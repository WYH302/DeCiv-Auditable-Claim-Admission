"""Pure deterministic semantic estimators for the isolated r12 channel."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from random import Random

from deciv2.r12.contracts import (
    SEMANTIC_STATUSES,
    SemanticEstimatorInput,
    SemanticPosterior,
    SemanticStatus,
    canonical_json_bytes,
    semantic_posterior,
)


MAJORITY_ESTIMATOR_ID = "semantic_majority_v1"
DAWID_SKENE_ESTIMATOR_ID = "semantic_dawid_skene_v1"
SOFTMAX_ESTIMATOR_ID = "semantic_softmax_v1"
SOFTMAX_SEEDS = (17, 29, 43)
SEMANTIC_SOFTMAX_FEATURES: tuple[str, ...] = (
    "vote_frequency_supported",
    "vote_frequency_contradicted",
    "vote_frequency_partially_supported",
    "vote_frequency_insufficient_evidence",
    "recorded_confidence_mass_supported",
    "recorded_confidence_mass_contradicted",
    "recorded_confidence_mass_partially_supported",
    "recorded_confidence_mass_insufficient_evidence",
    "recorded_confidence_coverage",
)

_STATE_SCHEMA = "deciv2.r12.semantic_estimator_state.v1"
_UNIFORM = (0.25, 0.25, 0.25, 0.25)
_DS_SMOOTHING = 0.001
_DS_MAXIMUM_ITERATIONS = 200
_DS_TOLERANCE = 1e-8
_SOFTMAX_EPOCHS = 80
_SOFTMAX_LEARNING_RATE = 0.15
_SOFTMAX_L2 = 0.01
_SOFTMAX_TOLERANCE = 1e-8
_SOFTMAX_FEATURE_FORMULAS = (
    "status_vote_count_divided_by_total_votes",
    "status_vote_count_divided_by_total_votes",
    "status_vote_count_divided_by_total_votes",
    "status_vote_count_divided_by_total_votes",
    "status_recorded_confidence_sum_divided_by_total_votes",
    "status_recorded_confidence_sum_divided_by_total_votes",
    "status_recorded_confidence_sum_divided_by_total_votes",
    "status_recorded_confidence_sum_divided_by_total_votes",
    "recorded_confidence_count_divided_by_total_votes",
)


def _validated_inputs(value: object) -> tuple[SemanticEstimatorInput, ...]:
    if (
        type(value) is not tuple
        or not value
        or any(not isinstance(item, SemanticEstimatorInput) for item in value)
    ):
        raise ValueError("inputs must be a nonempty tuple of SemanticEstimatorInput values")
    return value


def _validated_labels(value: object, count: int) -> tuple[SemanticStatus, ...]:
    if (
        type(value) is not tuple
        or not value
        or len(value) != count
        or any(type(label) is not str for label in value)
        or any(label not in SEMANTIC_STATUSES for label in value)
    ):
        raise ValueError("training_labels must be a matching nonempty tuple of semantic statuses")
    return value


def _state_hash(state: bytes) -> str:
    return hashlib.sha256(state).hexdigest().upper()


def _posterior(
    estimator_id: str,
    estimator_state_hash: str,
    semantic_input: SemanticEstimatorInput,
    values: tuple[float, ...],
) -> SemanticPosterior:
    return semantic_posterior(
        estimator_id=estimator_id,
        estimator_state_hash=estimator_state_hash,
        semantic_input=semantic_input,
        probabilities={status: values[index] for index, status in enumerate(SEMANTIC_STATUSES)},
    )


def semantic_softmax_feature_vector(
    semantic_input: SemanticEstimatorInput,
) -> tuple[float, ...]:
    """Return the frozen nine-feature direct semantic representation."""
    if not isinstance(semantic_input, SemanticEstimatorInput):
        raise ValueError("semantic_input must be SemanticEstimatorInput")
    votes = semantic_input.semantic_votes
    if not votes:
        return (0.0,) * len(SEMANTIC_SOFTMAX_FEATURES)
    count = len(votes)
    frequencies = tuple(
        sum(1 for vote in votes if vote.status == status) / count
        for status in SEMANTIC_STATUSES
    )
    confidence_masses = tuple(
        sum(vote.confidence for vote in votes if vote.status == status and vote.confidence is not None) / count
        for status in SEMANTIC_STATUSES
    )
    coverage = sum(1 for vote in votes if vote.confidence is not None) / count
    return frequencies + confidence_masses + (coverage,)


def _log_sum_exp(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _ds_expectation(
    rows: tuple[SemanticEstimatorInput, ...],
    validator_axis: tuple[str, ...],
    priors: tuple[float, ...],
    confusion: tuple[tuple[tuple[float, ...], ...], ...],
) -> tuple[tuple[tuple[float, ...], ...], float]:
    responsibilities: list[tuple[float, ...]] = []
    likelihood = 0.0
    for row in rows:
        scores = []
        for latent_index in range(len(SEMANTIC_STATUSES)):
            score = math.log(priors[latent_index])
            for vote in row.semantic_votes:
                if vote.validator_id in validator_axis:
                    validator_index = validator_axis.index(vote.validator_id)
                    observed_index = SEMANTIC_STATUSES.index(vote.status)
                    score += math.log(confusion[validator_index][latent_index][observed_index])
            scores.append(score)
        normalizer = _log_sum_exp(tuple(scores))
        likelihood += normalizer
        responsibilities.append(tuple(math.exp(score - normalizer) for score in scores))
    return tuple(responsibilities), likelihood


def _ds_maximization(
    rows: tuple[SemanticEstimatorInput, ...],
    validator_axis: tuple[str, ...],
    responsibilities: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], tuple[tuple[tuple[float, ...], ...], ...]]:
    priors = tuple(
        sum(row[latent_index] for row in responsibilities) / len(rows)
        for latent_index in range(len(SEMANTIC_STATUSES))
    )
    counts = [
        [
            [_DS_SMOOTHING for _ in SEMANTIC_STATUSES]
            for _ in SEMANTIC_STATUSES
        ]
        for _ in validator_axis
    ]
    for semantic_input, row_responsibilities in zip(rows, responsibilities, strict=True):
        for vote in semantic_input.semantic_votes:
            validator_index = validator_axis.index(vote.validator_id)
            observed_index = SEMANTIC_STATUSES.index(vote.status)
            for latent_index, responsibility in enumerate(row_responsibilities):
                counts[validator_index][latent_index][observed_index] += responsibility
    confusion = tuple(
        tuple(
            tuple(value / sum(latent_counts) for value in latent_counts)
            for latent_counts in validator_counts
        )
        for validator_counts in counts
    )
    return priors, confusion


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(values)
    unnormalized = tuple(math.exp(value - maximum) for value in values)
    total = sum(unnormalized)
    return tuple(value / total for value in unnormalized)


def _softmax_logits(
    row: tuple[float, ...], weights: tuple[tuple[float, ...], ...]
) -> tuple[float, ...]:
    return tuple(
        class_weights[0]
        + sum(coefficient * value for coefficient, value in zip(class_weights[1:], row, strict=True))
        for class_weights in weights
    )


def _standardize(
    matrix: tuple[tuple[float, ...], ...],
    means: tuple[float, ...],
    scales: tuple[float, ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple((value - mean) / scale for value, mean, scale in zip(row, means, scales, strict=True))
        for row in matrix
    )


def _softmax_objective(
    matrix: tuple[tuple[float, ...], ...],
    label_indices: tuple[int, ...],
    weights: tuple[tuple[float, ...], ...],
) -> float:
    total = 0.0
    for row, label_index in zip(matrix, label_indices, strict=True):
        logits = _softmax_logits(row, weights)
        maximum = max(logits)
        total += maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[label_index]
    penalty = 0.5 * _SOFTMAX_L2 * sum(value * value for class_weights in weights for value in class_weights[1:])
    return total / len(matrix) + penalty


@dataclass(frozen=True)
class SemanticMajorityEstimator:
    """Canonical vote-frequency semantic estimator."""

    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        training_inputs: tuple[SemanticEstimatorInput, ...],
    ) -> SemanticMajorityEstimator:
        _validated_inputs(training_inputs)
        fitted = SemanticMajorityEstimator()
        object.__setattr__(fitted, "_fitted", True)
        return fitted

    def predict_proba(
        self,
        inputs: tuple[SemanticEstimatorInput, ...],
    ) -> tuple[SemanticPosterior, ...]:
        if not self._fitted:
            raise ValueError("SemanticMajorityEstimator must be fitted before prediction")
        values = _validated_inputs(inputs)
        state_hash = self.estimator_state_hash
        outputs = []
        for semantic_input in values:
            vote_count = len(semantic_input.semantic_votes)
            row = (
                tuple(
                    sum(1 for vote in semantic_input.semantic_votes if vote.status == status) / vote_count
                    for status in SEMANTIC_STATUSES
                )
                if vote_count
                else _UNIFORM
            )
            outputs.append(_posterior(MAJORITY_ESTIMATOR_ID, state_hash, semantic_input, row))
        return tuple(outputs)

    def state_bytes(self) -> bytes:
        if not self._fitted:
            raise ValueError("unfitted estimator state is not serializable")
        payload = {
            "state_schema": _STATE_SCHEMA,
            "estimator_id": MAJORITY_ESTIMATOR_ID,
            "semantic_axis": SEMANTIC_STATUSES,
            "algorithm": "canonical_vote_frequency",
            "zero_vote_policy": "uniform_semantic_axis",
        }
        return canonical_json_bytes("semantic_estimator_state", payload)

    @property
    def estimator_state_hash(self) -> str:
        return _state_hash(self.state_bytes())


@dataclass(frozen=True)
class SemanticDawidSkeneEstimator:
    """Four-class unsupervised Dawid--Skene estimator."""

    validator_axis: tuple[str, ...] = field(default=(), init=False)
    priors: tuple[float, ...] = field(default=_UNIFORM, init=False)
    confusion: tuple[tuple[tuple[float, ...], ...], ...] = field(default=(), init=False)
    iterations: int = field(default=0, init=False)
    final_log_likelihood: float = field(default=0.0, init=False)
    converged: bool = field(default=False, init=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(
        self,
        training_inputs: tuple[SemanticEstimatorInput, ...],
    ) -> SemanticDawidSkeneEstimator:
        validated = _validated_inputs(training_inputs)
        rows = tuple(
            sorted(
                validated,
                key=lambda row: tuple((vote.validator_id, vote.status) for vote in row.semantic_votes),
            )
        )
        validator_axis = tuple(sorted({vote.validator_id for row in rows for vote in row.semantic_votes}))
        priors = _UNIFORM
        confusion = tuple(
            tuple(
                tuple(0.7 if latent_index == observed_index else 0.1 for observed_index in range(len(SEMANTIC_STATUSES)))
                for latent_index in range(len(SEMANTIC_STATUSES))
            )
            for _ in validator_axis
        )
        final_log_likelihood = float("-inf")
        converged = False
        iterations = 0
        for iterations in range(1, _DS_MAXIMUM_ITERATIONS + 1):
            responsibilities, previous_likelihood = _ds_expectation(rows, validator_axis, priors, confusion)
            priors, confusion = _ds_maximization(rows, validator_axis, responsibilities)
            _, final_log_likelihood = _ds_expectation(rows, validator_axis, priors, confusion)
            if abs(final_log_likelihood - previous_likelihood) < _DS_TOLERANCE:
                converged = True
                break
        fitted = SemanticDawidSkeneEstimator()
        object.__setattr__(fitted, "validator_axis", validator_axis)
        object.__setattr__(fitted, "priors", priors)
        object.__setattr__(fitted, "confusion", confusion)
        object.__setattr__(fitted, "iterations", iterations)
        object.__setattr__(fitted, "final_log_likelihood", final_log_likelihood)
        object.__setattr__(fitted, "converged", converged)
        object.__setattr__(fitted, "_fitted", True)
        return fitted

    def predict_proba(
        self,
        inputs: tuple[SemanticEstimatorInput, ...],
    ) -> tuple[SemanticPosterior, ...]:
        if not self._fitted:
            raise ValueError("SemanticDawidSkeneEstimator must be fitted before prediction")
        values = _validated_inputs(inputs)
        responsibilities = tuple(
            self.priors
            if not any(vote.validator_id in self.validator_axis for vote in semantic_input.semantic_votes)
            else _ds_expectation((semantic_input,), self.validator_axis, self.priors, self.confusion)[0][0]
            for semantic_input in values
        )
        state_hash = self.estimator_state_hash
        return tuple(
            _posterior(DAWID_SKENE_ESTIMATOR_ID, state_hash, semantic_input, row)
            for semantic_input, row in zip(values, responsibilities, strict=True)
        )

    def state_bytes(self) -> bytes:
        if not self._fitted:
            raise ValueError("unfitted estimator state is not serializable")
        payload = {
            "state_schema": _STATE_SCHEMA,
            "estimator_id": DAWID_SKENE_ESTIMATOR_ID,
            "semantic_axis": SEMANTIC_STATUSES,
            "algorithm": "four_class_dawid_skene_em",
            "parameters": {
                "initial_priors": _UNIFORM,
                "initial_confusion_diagonal": 0.7,
                "initial_confusion_off_diagonal": 0.1,
                "confusion_smoothing": _DS_SMOOTHING,
                "maximum_iterations": _DS_MAXIMUM_ITERATIONS,
                "convergence_tolerance": _DS_TOLERANCE,
                "convergence_rule": "absolute_updated_minus_previous_log_likelihood",
            },
            "fitted_state": {
                "validator_axis": self.validator_axis,
                "priors": self.priors,
                "confusion": self.confusion,
                "iterations": self.iterations,
                "final_log_likelihood": self.final_log_likelihood,
                "converged": self.converged,
            },
            "prediction_policy": {
                "unseen_validator": "ignore_likelihood_term",
                "zero_vote": "fitted_priors",
            },
        }
        return canonical_json_bytes("semantic_estimator_state", payload)

    @property
    def estimator_state_hash(self) -> str:
        return _state_hash(self.state_bytes())


@dataclass(frozen=True)
class SemanticSoftmaxEstimator:
    """Direct four-class softmax regression over the frozen semantic features."""

    seed: int
    means: tuple[float, ...] = field(default=(), init=False)
    scales: tuple[float, ...] = field(default=(), init=False)
    weights: tuple[tuple[float, ...], ...] = field(default=(), init=False)
    loss_history: tuple[float, ...] = field(default=(), init=False)
    iterations: int = field(default=0, init=False)
    converged: bool = field(default=False, init=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in SOFTMAX_SEEDS:
            raise ValueError("seed must be a non-boolean integer in SOFTMAX_SEEDS")

    def fit(
        self,
        training_inputs: tuple[SemanticEstimatorInput, ...],
        training_labels: tuple[SemanticStatus, ...],
    ) -> SemanticSoftmaxEstimator:
        validated_inputs = _validated_inputs(training_inputs)
        validated_labels = _validated_labels(training_labels, len(validated_inputs))
        pairs = tuple(
            sorted(
                zip(
                    (semantic_softmax_feature_vector(item) for item in validated_inputs),
                    validated_labels,
                    strict=True,
                )
            )
        )
        matrix = tuple(pair[0] for pair in pairs)
        labels = tuple(pair[1] for pair in pairs)
        means = tuple(
            sum(row[index] for row in matrix) / len(matrix)
            for index in range(len(SEMANTIC_SOFTMAX_FEATURES))
        )
        raw_scales = tuple(
            math.sqrt(sum((row[index] - means[index]) ** 2 for row in matrix) / len(matrix))
            for index in range(len(SEMANTIC_SOFTMAX_FEATURES))
        )
        scales = tuple(
            scale if scale > 0.0 and math.isfinite(scale) else 1.0
            for scale in raw_scales
        )
        standardized = _standardize(matrix, means, scales)
        label_indices = tuple(SEMANTIC_STATUSES.index(label) for label in labels)
        generator = Random(self.seed)
        weights = tuple(
            tuple(generator.uniform(-0.001, 0.001) for _ in range(len(SEMANTIC_SOFTMAX_FEATURES) + 1))
            for _ in SEMANTIC_STATUSES
        )
        history: list[float] = []
        previous_loss: float | None = None
        converged = False
        for _ in range(_SOFTMAX_EPOCHS):
            gradients = [
                [0.0 for _ in range(len(SEMANTIC_SOFTMAX_FEATURES) + 1)]
                for _ in SEMANTIC_STATUSES
            ]
            for row, label_index in zip(standardized, label_indices, strict=True):
                row_probabilities = _softmax(_softmax_logits(row, weights))
                for class_index, probability in enumerate(row_probabilities):
                    residual = probability - (1.0 if class_index == label_index else 0.0)
                    gradients[class_index][0] += residual
                    for feature_index, value in enumerate(row, start=1):
                        gradients[class_index][feature_index] += residual * value
            weights = tuple(
                tuple(
                    value
                    - _SOFTMAX_LEARNING_RATE
                    * (
                        gradients[class_index][weight_index] / len(standardized)
                        + (_SOFTMAX_L2 * value if weight_index else 0.0)
                    )
                    for weight_index, value in enumerate(class_weights)
                )
                for class_index, class_weights in enumerate(weights)
            )
            loss = _softmax_objective(standardized, label_indices, weights)
            history.append(loss)
            if previous_loss is not None and abs(loss - previous_loss) < _SOFTMAX_TOLERANCE:
                converged = True
                break
            previous_loss = loss
        fitted = SemanticSoftmaxEstimator(self.seed)
        object.__setattr__(fitted, "means", means)
        object.__setattr__(fitted, "scales", scales)
        object.__setattr__(fitted, "weights", weights)
        object.__setattr__(fitted, "loss_history", tuple(history))
        object.__setattr__(fitted, "iterations", len(history))
        object.__setattr__(fitted, "converged", converged)
        object.__setattr__(fitted, "_fitted", True)
        return fitted

    def predict_proba(
        self,
        inputs: tuple[SemanticEstimatorInput, ...],
    ) -> tuple[SemanticPosterior, ...]:
        if not self._fitted:
            raise ValueError("SemanticSoftmaxEstimator must be fitted before prediction")
        values = _validated_inputs(inputs)
        matrix = tuple(semantic_softmax_feature_vector(item) for item in values)
        standardized = _standardize(matrix, self.means, self.scales)
        state_hash = self.estimator_state_hash
        return tuple(
            _posterior(SOFTMAX_ESTIMATOR_ID, state_hash, semantic_input, _softmax(_softmax_logits(row, self.weights)))
            for semantic_input, row in zip(values, standardized, strict=True)
        )

    def state_bytes(self) -> bytes:
        if not self._fitted:
            raise ValueError("unfitted estimator state is not serializable")
        payload = {
            "state_schema": _STATE_SCHEMA,
            "estimator_id": SOFTMAX_ESTIMATOR_ID,
            "semantic_axis": SEMANTIC_STATUSES,
            "algorithm": "direct_multinomial_softmax_regression",
            "feature_axis": SEMANTIC_SOFTMAX_FEATURES,
            "feature_formulas": _SOFTMAX_FEATURE_FORMULAS,
            "parameters": {
                "seed": self.seed,
                "epochs": _SOFTMAX_EPOCHS,
                "learning_rate": _SOFTMAX_LEARNING_RATE,
                "l2": _SOFTMAX_L2,
                "tolerance": _SOFTMAX_TOLERANCE,
                "initialization": "python_random_uniform",
                "initialization_interval": (-0.001, 0.001),
                "standardization": "training_population_mean_std_zero_or_nonfinite_to_one",
                "optimizer": "full_batch_gradient_descent",
                "objective": "mean_cross_entropy_plus_half_l2_non_intercept_squared_weights",
                "stopping_rule": "absolute_consecutive_post_update_loss_difference",
            },
            "fitted_state": {
                "means": self.means,
                "scales": self.scales,
                "weights": self.weights,
                "loss_history": self.loss_history,
                "iterations": self.iterations,
                "converged": self.converged,
            },
        }
        return canonical_json_bytes("semantic_estimator_state", payload)

    @property
    def estimator_state_hash(self) -> str:
        return _state_hash(self.state_bytes())


__all__ = [
    "MAJORITY_ESTIMATOR_ID",
    "DAWID_SKENE_ESTIMATOR_ID",
    "SOFTMAX_ESTIMATOR_ID",
    "SOFTMAX_SEEDS",
    "SEMANTIC_SOFTMAX_FEATURES",
    "semantic_softmax_feature_vector",
    "SemanticMajorityEstimator",
    "SemanticDawidSkeneEstimator",
    "SemanticSoftmaxEstimator",
]
