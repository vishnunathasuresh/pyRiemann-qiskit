import numpy as np
import pytest
from pyriemann.estimation import XdawnCovariances
from pyriemann.utils.distance import distance_logeuclid
from pyriemann.utils.mean import mean_logeuclid
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

from pyriemann_qiskit.classification import QuanticMDM
from pyriemann_qiskit.optimization.distance import (
    qdistance_logeuclid_to_convex_hull,
    weights_logeuclid_to_convex_hull,
)
from pyriemann_qiskit.optimization.docplex import ClassicalOptimizer
from pyriemann_qiskit.optimization.simplex import (
    SingleExcitationHullOptimizer,
    _decode_counts,
    _single_excitation_circuit,
    _uniform_angles,
)
from pyriemann_qiskit.utils.dataset import get_mne_sample


@pytest.mark.parametrize(
    "metric",
    [
        {"mean": "euclid", "distance": "qeuclid"},
        {"mean": "logeuclid", "distance": "qlogeuclid"},
        {"mean": "logeuclid", "distance": "qlogeuclid_hull"},
    ],
)
def test_performance(metric):
    clf = make_pipeline(XdawnCovariances(), QuanticMDM(metric=metric, quantum=False))
    skf = StratifiedKFold(n_splits=3)
    covset, labels = get_mne_sample()
    score = cross_val_score(clf, covset, labels, cv=skf, scoring="roc_auc")
    assert score.mean() > 0


@pytest.mark.parametrize(
    "optimizer",
    [
        ClassicalOptimizer(),
        # NaiveQAOAOptimizer(),
        # QAOACVOptimizer()
    ],
)
def test_qdistance_logeuclid_to_convex_hull(optimizer, get_covmats):
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)

    dist = qdistance_logeuclid_to_convex_hull(covmats, covmats[0], optimizer=optimizer)
    assert dist == pytest.approx(0, rel=1e-5, abs=1e-5)

    covmean = mean_logeuclid(covmats)
    dist = qdistance_logeuclid_to_convex_hull(covmats, covmean, optimizer=optimizer)
    assert dist == pytest.approx(0, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "optimizer",
    [
        ClassicalOptimizer(),
        # NaiveQAOAOptimizer(),
        # QAOACVOptimizer()
    ],
)
def test_weight_logeuclid_to_convex_hull(optimizer):
    X_0 = np.array([[0.9, 1.1], [0.9, 1.1]])
    X_1 = X_0 + 1
    X_train = np.stack((X_0, X_1))
    X_test = (X_0 + X_1) / 3
    weights = weights_logeuclid_to_convex_hull(X_train, X_test, optimizer=optimizer)
    distances = 1 - weights
    assert distances.argmin() == 0


def test_single_excitation_circuit_prepares_vertex_and_uniform_simplex():
    """Every ideal outcome has exactly one excited qubit."""
    from qiskit.quantum_info import Statevector

    circuit, parameters = _single_excitation_circuit(5)
    for angles, expected in (
        (np.zeros(4), [1, 0, 0, 0, 0]),
        (_uniform_angles(5), [0.2] * 5),
    ):
        probabilities = Statevector(
            circuit.assign_parameters(dict(zip(parameters, angles)))
        ).probabilities()
        decoded = np.array([probabilities[1 << index] for index in range(5)])

        np.testing.assert_allclose(decoded, expected, atol=1e-12)
        assert sum(
            probability
            for bitstring, probability in enumerate(probabilities)
            if bitstring.bit_count() != 1
        ) == pytest.approx(0, abs=1e-12)


def test_single_excitation_circuit_spans_arbitrary_simplex_weights():
    """Analytical exchange angles recover a nonuniform simplex point."""
    from qiskit.quantum_info import Statevector

    expected = np.array([0.37, 0.21, 0.19, 0.14, 0.09])
    angles = []
    remaining = 1.0
    for weight in expected[:-1]:
        angles.append(2 * np.arccos(np.sqrt(weight / remaining)))
        remaining -= weight

    circuit, parameters = _single_excitation_circuit(len(expected))
    probabilities = Statevector(
        circuit.assign_parameters(dict(zip(parameters, angles)))
    ).probabilities()
    decoded = np.array([probabilities[1 << index] for index in range(len(expected))])

    np.testing.assert_allclose(decoded, expected, atol=1e-12)
    assert np.sum(decoded) == pytest.approx(1, abs=1e-12)


def test_single_excitation_decoder_rejects_insufficient_valid_shots():
    """Invalid bitstrings are excluded and low valid counts fail explicitly."""
    counts = {"001": 60, "010": 20, "100": 10, "000": 10}
    weights, fraction = _decode_counts(counts, 3, 0.5, 32)
    np.testing.assert_allclose(weights, [60 / 90, 20 / 90, 10 / 90])
    assert fraction == pytest.approx(0.9)

    weights, fraction = _decode_counts(counts, 3, 0.95, 32)
    assert weights is None
    assert fraction == pytest.approx(0.9)
    weights, fraction = _decode_counts({"000": 100}, 3, 0.5, 32)
    assert weights is None
    assert fraction == 0


def test_single_excitation_decoder_uses_qiskit_bit_order():
    """The leftmost displayed bit maps to the last weight index."""
    counts = {"00100": 37, "00010": 21, "00001": 42, "11000": 5}
    weights, fraction = _decode_counts(counts, 5, 0.5, 32)
    np.testing.assert_allclose(weights, [0.42, 0.21, 0.37, 0, 0])
    assert fraction == pytest.approx(100 / 105)


def test_single_excitation_hull_fails_without_reliable_evaluation(monkeypatch):
    """An invalid-shot run cannot return a plausible classical fallback."""
    import pyriemann_qiskit.optimization.simplex as simplex_module
    from pyriemann_qiskit.optimization.cobyla_optimizer import CobylaOptimizer

    monkeypatch.setattr(simplex_module, "_decode_counts", lambda *args: (None, 0))
    matrices = np.array([[[1.0]], [[9.0]]])
    optimizer = SingleExcitationHullOptimizer(
        exact=True, optimizer=CobylaOptimizer(maxiter=3)
    )
    with pytest.raises(RuntimeError, match="enough one-excitation shots"):
        weights_logeuclid_to_convex_hull(matrices, np.array([[3.0]]), optimizer)


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (1.0, [1.0, 0.0]),
        (3.0, [0.5, 0.5]),
    ],
)
def test_single_excitation_hull_exact_is_feasible_and_repeatable(target, expected):
    """The hull loss returns the same feasible weights it optimized."""
    from pyriemann_qiskit.optimization.cobyla_optimizer import CobylaOptimizer

    matrices = np.array([[[1.0]], [[9.0]]])
    target_matrix = np.array([[target]])
    solutions = []
    for _ in range(2):
        optimizer = SingleExcitationHullOptimizer(
            exact=True,
            optimizer=CobylaOptimizer(maxiter=150),
            min_valid_fraction=1.0,
        )
        weights = weights_logeuclid_to_convex_hull(matrices, target_matrix, optimizer)
        solutions.append(weights)
        np.testing.assert_array_equal(weights, optimizer.weights_)
        assert weights.sum() == pytest.approx(1, abs=1e-12)
        assert np.all(weights >= 0)
        assert optimizer.minimum_ == pytest.approx(
            distance_logeuclid(mean_logeuclid(matrices, weights), target_matrix) ** 2,
            abs=1e-10,
        )
        assert optimizer.valid_fraction_ == pytest.approx(1, abs=1e-12)
        assert optimizer.evaluations_ > 0
    np.testing.assert_array_equal(solutions[0], solutions[1])
    np.testing.assert_allclose(solutions[0], expected, atol=1e-3)


def test_single_excitation_hull_shot_sampler_is_feasible():
    """The sampled backend decodes only one-excitation outcomes."""
    from pyriemann_qiskit.optimization.cobyla_optimizer import CobylaOptimizer

    matrices = np.array([[[1.0]], [[9.0]]])
    optimizer = SingleExcitationHullOptimizer(
        optimizer=CobylaOptimizer(maxiter=5), shots=128, seed=42
    )
    weights = weights_logeuclid_to_convex_hull(matrices, np.array([[3.0]]), optimizer)

    np.testing.assert_array_equal(weights, optimizer.weights_)
    assert weights.sum() == pytest.approx(1, abs=1e-12)
    assert np.all(weights >= 0)
    assert optimizer.valid_fraction_ == pytest.approx(1, abs=1e-12)
