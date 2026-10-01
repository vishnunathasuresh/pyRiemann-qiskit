"""Single-excitation variational optimization for Log-Euclidean hulls."""

import time

import numpy as np
from pyriemann.utils.base import logm
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.primitives import BackendSamplerV2
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_algorithms.optimizers import SPSA

from ..utils.quantum_provider import get_simulator


def _single_excitation_circuit(n_weights):
    parameters = list(ParameterVector("theta", n_weights - 1))
    circuit = QuantumCircuit(n_weights)
    circuit.x(0)
    for index, parameter in enumerate(parameters):
        circuit.append(XXPlusYYGate(parameter), [index, index + 1])
    return circuit, parameters


def _uniform_angles(n_weights):
    return np.array(
        [
            2 * np.arccos(np.sqrt(1 / (n_weights - index)))
            for index in range(n_weights - 1)
        ]
    )


def _decode_counts(counts, n_weights, min_valid_fraction, min_valid_shots):
    weights = np.zeros(n_weights)
    total = sum(counts.values())
    if total <= 0:
        return None, 0.0
    for bitstring, count in counts.items():
        bits = int(bitstring.replace(" ", ""), 2)
        if bits.bit_count() == 1:
            weights[bits.bit_length() - 1] += count
    valid = float(np.sum(weights))
    fraction = valid / total
    if valid < min_valid_shots or fraction + 1e-12 < min_valid_fraction:
        return None, fraction
    return weights / valid, fraction


class SingleExcitationHullOptimizer:
    """Optimize Log-Euclidean hull weights in a one-excitation state.

    The circuit contains one excited qubit and ``n_matrices - 1`` exchange
    rotations. Its ideal measurement probabilities are simplex weights. In
    the chain, ``w[0] = cos(theta[0] / 2)**2`` and each later weight is the
    remaining probability multiplied by ``cos(theta[i] / 2)**2``. Given any
    simplex vector, setting ``theta[i] = 2*arccos(sqrt(w[i] / remaining))``
    recursively represents it, with an arbitrary later angle after the
    remaining probability reaches zero. This establishes coverage of the
    full simplex with ``n_matrices - 1`` angles.

    In shot mode, only measured bitstrings with exactly one ``1`` contribute to
    the weights. Too few valid shots mark an evaluation as unreliable; if no
    reliable evaluation occurs, :meth:`solve_hull` raises ``RuntimeError``.

    Parameters
    ----------
    optimizer : qiskit_algorithms.optimizers.Optimizer or None, default=None
        Classical optimizer for circuit angles. ``None`` uses SPSA with 125
        iterations.
    shots : int, default=1024
        Measurement shots per loss evaluation in shot mode.
    exact : bool, default=False
        Use exact statevector probabilities instead of sampling. This is
        useful for reproducible small simulations.
    quantum_instance : BackendSamplerV2 or None, default=None
        Sampler for shot mode. ``None`` uses the configured local simulator.
    seed : int, default=42
        Seed for the default local sampler.
    min_valid_fraction : float, default=0.5
        Minimum fraction of one-excitation shots for a reliable evaluation.
    min_valid_shots : int, default=32
        Minimum count of one-excitation shots for a reliable evaluation.

    Attributes
    ----------
    weights_ : ndarray
        Returned feasible weights from the best loss evaluation.
    valid_fraction_ : float
        Fraction of valid shots in that evaluation.
    optim_params_ : ndarray
        Circuit parameters of that evaluation.
    minimum_ : float
        Squared Log-Euclidean distance of the returned weights.
    evaluations_ : int
        Number of loss evaluations.
    run_time_ : float
        Optimization time in seconds.
    """

    def __init__(
        self,
        optimizer=None,
        shots=1024,
        exact=False,
        quantum_instance=None,
        seed=42,
        min_valid_fraction=0.5,
        min_valid_shots=32,
    ):
        if shots < 1 or min_valid_shots < 1:
            raise ValueError("shots and min_valid_shots must be positive")
        if not exact and min_valid_shots > shots:
            raise ValueError("min_valid_shots cannot exceed shots")
        if not 0 < min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in (0, 1]")
        self.optimizer = SPSA(maxiter=125) if optimizer is None else optimizer
        self.shots = shots
        self.exact = exact
        self.quantum_instance = quantum_instance
        self.seed = seed
        self.min_valid_fraction = min_valid_fraction
        self.min_valid_shots = min_valid_shots

    def solve_hull(self, matrices, target):
        """Return simplex weights minimizing Log-Euclidean hull distance.

        Parameters
        ----------
        matrices : ndarray, shape (n_matrices, n_channels, n_channels)
            SPD hull elements.
        target : ndarray, shape (n_channels, n_channels)
            SPD target matrix.

        Returns
        -------
        weights : ndarray, shape (n_matrices,)
            Nonnegative weights whose sum is one.
        """
        n_weights = len(matrices)
        if n_weights < 2:
            raise ValueError("single-excitation hull needs at least two matrices")
        logs = np.array([logm(matrix) for matrix in matrices])
        target_log = logm(target)
        circuit, parameters = _single_excitation_circuit(n_weights)
        if self.exact:
            measured_circuit = None
            sampler = None
        else:
            sampler = self.quantum_instance
            if sampler is None:
                sampler = BackendSamplerV2(
                    backend=get_simulator(),
                    options={
                        "default_shots": self.shots,
                        "seed_simulator": self.seed,
                    },
                )
            pass_manager = generate_preset_pass_manager(
                optimization_level=1, backend=sampler.backend
            )
            measured_circuit = circuit.copy()
            measured_circuit.measure_all()
            measured_circuit = pass_manager.run(measured_circuit)

        self.evaluations_ = 0
        best = None

        def loss(angles):
            nonlocal best
            binding = dict(zip(parameters, angles))
            if self.exact:
                probabilities = Statevector(
                    circuit.assign_parameters(binding)
                ).probabilities()
                counts = {
                    format(index, f"0{n_weights}b"): probability
                    for index, probability in enumerate(probabilities)
                    if probability > 0
                }
                weights, valid_fraction = _decode_counts(
                    counts, n_weights, self.min_valid_fraction, 0
                )
            else:
                bound = measured_circuit.assign_parameters(binding)
                job = sampler.run([bound], shots=self.shots)
                counts = job.result()[0].data.meas.get_counts()
                weights, valid_fraction = _decode_counts(
                    counts,
                    n_weights,
                    self.min_valid_fraction,
                    self.min_valid_shots,
                )
            self.evaluations_ += 1
            if weights is None:
                return 1e6
            residual = np.tensordot(weights, logs, axes=(0, 0)) - target_log
            value = float(np.sum(residual**2))
            if best is None or value < best[0]:
                best = (
                    value,
                    weights.copy(),
                    valid_fraction,
                    np.asarray(angles).copy(),
                )
            return value

        start = time.perf_counter()
        self.optimizer.minimize(loss, _uniform_angles(n_weights), bounds=None)
        self.run_time_ = time.perf_counter() - start
        if best is None:
            raise RuntimeError("no evaluation had enough one-excitation shots")
        self.minimum_, self.weights_, self.valid_fraction_, self.optim_params_ = best
        return self.weights_.copy()
