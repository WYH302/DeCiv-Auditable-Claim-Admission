"""Immutable, hash-bound r12 semantic and governance boundary objects."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping, TypeAlias


DOMAIN_VERSION = "deciv2.r12.v1"
SEMANTIC_STATUSES = (
    "supported",
    "contradicted",
    "partially_supported",
    "insufficient_evidence",
)
ADMISSION_ACTIONS = ("accept", "reject", "revise", "defer")
PROVENANCE_STATUSES = ("complete", "incomplete", "unknown")
AUDITABILITY_STATUSES = ("auditable", "unauditable", "unknown")
REPAIRABILITY_STATUSES = ("verified_repair", "no_repair", "unknown")
ELIGIBILITY_REASON_CODES = (
    "accept_provenance_not_allowed",
    "accept_auditability_not_allowed",
    "accept_human_review_required",
    "revise_repair_not_verified",
)
SemanticStatus: TypeAlias = str
AdmissionAction: TypeAlias = str
_HASH = re.compile(r"^[A-F0-9]{64}$")


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _hash(value: object, name: str) -> str:
    value = _identifier(value, name)
    if not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be an uppercase SHA-256 digest")
    return value


def _bounded(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite float in [0, 1]")
    return number


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be finite")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _exact_mapping(value: object, keys: tuple[str, ...], name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must have exactly {keys}")
    actual_keys = tuple(value)
    if any(type(key) is not str for key in actual_keys) or set(actual_keys) != set(keys):
        raise ValueError(f"{name} must have exactly {keys}")
    return value


def _axis(value: object, allowed: tuple[str, ...], name: str) -> str:
    value = _identifier(value, name)
    if value not in allowed:
        raise ValueError(f"unknown {name}: {value}")
    return value


def _normalise(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("hash payload contains a non-finite float")
        return (0.0 if value == 0.0 else value).hex()
    if isinstance(value, tuple) or isinstance(value, list):
        return [_normalise(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise ValueError(f"unsupported canonical payload value: {type(value).__name__}")


def canonical_json_bytes(record_type: str, payload: Mapping[str, object]) -> bytes:
    """Return the exact versioned canonical JSON preimage used by all r12 hashes."""
    envelope = {"schema": DOMAIN_VERSION, "type": _identifier(record_type, "record_type"), "payload": _normalise(payload)}
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(record_type: str, payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(record_type, payload)).hexdigest().upper()


def _derived_or_verified(supplied: object, record_type: str, payload: Mapping[str, object], name: str) -> str:
    derived = _digest(record_type, payload)
    if supplied is None:
        return derived
    if type(supplied) is not str:
        raise ValueError(f"{name} must be an exact string")
    if supplied == "":
        return derived
    if _hash(supplied, name) != derived:
        raise ValueError(f"{name} does not match its canonical preimage")
    return derived


@dataclass(frozen=True)
class SemanticObservations:
    content_support_score: float
    contradiction_score: float
    partial_support_score: float
    uncertainty_score: float

    def __post_init__(self) -> None:
        for name in ("content_support_score", "contradiction_score", "partial_support_score", "uncertainty_score"):
            object.__setattr__(self, name, _bounded(getattr(self, name), name))

    def payload(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in ("content_support_score", "contradiction_score", "partial_support_score", "uncertainty_score")}


@dataclass(frozen=True)
class SemanticVote:
    validator_id: str
    status: SemanticStatus
    confidence: float | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "validator_id", _identifier(self.validator_id, "validator_id"))
        object.__setattr__(self, "status", _axis(self.status, SEMANTIC_STATUSES, "semantic status"))
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _bounded(self.confidence, "confidence"))

    def payload(self) -> dict[str, object]:
        return {"validator_id": self.validator_id, "status": self.status, "confidence": self.confidence}


@dataclass(frozen=True)
class SemanticEstimatorInput:
    observations: SemanticObservations
    semantic_votes: tuple[SemanticVote, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observations, SemanticObservations):
            raise ValueError("observations must be SemanticObservations")
        if not isinstance(self.semantic_votes, tuple) or any(not isinstance(vote, SemanticVote) for vote in self.semantic_votes):
            raise ValueError("semantic_votes must be a tuple of SemanticVote values")
        ordered = tuple(sorted(self.semantic_votes, key=lambda vote: vote.validator_id))
        if len({vote.validator_id for vote in ordered}) != len(ordered):
            raise ValueError("validator_id values must be unique")
        object.__setattr__(self, "semantic_votes", ordered)

    def payload(self) -> dict[str, object]:
        return {"observations": self.observations.payload(), "semantic_votes": [vote.payload() for vote in self.semantic_votes]}

    @property
    def semantic_input_hash(self) -> str:
        return _digest("semantic_input", self.payload())


@dataclass(frozen=True)
class SemanticPosterior:
    estimator_id: str
    estimator_state_hash: str
    semantic_input_hash: str
    probabilities: tuple[tuple[SemanticStatus, float], ...]
    posterior_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimator_id", _identifier(self.estimator_id, "estimator_id"))
        object.__setattr__(self, "estimator_state_hash", _hash(self.estimator_state_hash, "estimator_state_hash"))
        object.__setattr__(self, "semantic_input_hash", _hash(self.semantic_input_hash, "semantic_input_hash"))
        if not isinstance(self.probabilities, tuple) or len(self.probabilities) != len(SEMANTIC_STATUSES):
            raise ValueError("probabilities must contain each semantic status exactly once")
        if any(not isinstance(item, tuple) or len(item) != 2 for item in self.probabilities):
            raise ValueError("probabilities must contain status/probability pairs")
        statuses = tuple(_axis(item[0], SEMANTIC_STATUSES, "semantic status") for item in self.probabilities)
        if statuses != SEMANTIC_STATUSES:
            raise ValueError("probabilities must use canonical semantic-status order")
        probabilities = tuple((status, _bounded(probability, f"probability[{status}]")) for status, probability in self.probabilities)
        if abs(sum(probability for _, probability in probabilities) - 1.0) > 1e-12:
            raise ValueError("probabilities must sum to one within 1e-12")
        object.__setattr__(self, "probabilities", probabilities)
        payload = {"estimator_id": self.estimator_id, "estimator_state_hash": self.estimator_state_hash, "semantic_input_hash": self.semantic_input_hash, "probabilities": probabilities}
        object.__setattr__(self, "posterior_hash", _derived_or_verified(self.posterior_hash, "semantic_posterior", payload, "posterior_hash"))

    def probability(self, status: SemanticStatus) -> float:
        status = _axis(status, SEMANTIC_STATUSES, "semantic status")
        return self.probabilities[SEMANTIC_STATUSES.index(status)][1]


@dataclass(frozen=True)
class GovernanceContext:
    provenance_status: str
    evidence_auditability: str
    human_review_required: bool
    repairability: str
    policy_profile_id: str
    context_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance_status", _axis(self.provenance_status, PROVENANCE_STATUSES, "provenance_status"))
        object.__setattr__(self, "evidence_auditability", _axis(self.evidence_auditability, AUDITABILITY_STATUSES, "evidence_auditability"))
        if not isinstance(self.human_review_required, bool):
            raise ValueError("human_review_required must be boolean")
        object.__setattr__(self, "repairability", _axis(self.repairability, REPAIRABILITY_STATUSES, "repairability"))
        object.__setattr__(self, "policy_profile_id", _identifier(self.policy_profile_id, "policy_profile_id"))
        payload = self.payload()
        object.__setattr__(self, "context_hash", _derived_or_verified(self.context_hash, "governance_context", payload, "context_hash"))

    def payload(self) -> dict[str, object]:
        return {"provenance_status": self.provenance_status, "evidence_auditability": self.evidence_auditability, "human_review_required": self.human_review_required, "repairability": self.repairability, "policy_profile_id": self.policy_profile_id}


@dataclass(frozen=True)
class GovernanceProfile:
    profile_id: str
    base_costs: tuple[tuple[AdmissionAction, tuple[tuple[SemanticStatus, float], ...]], ...]
    accept_provenance_statuses: tuple[str, ...]
    accept_auditability_statuses: tuple[str, ...]
    allow_accept_when_human_review_required: bool
    profile_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, "profile_id"))
        if not isinstance(self.base_costs, tuple) or len(self.base_costs) != len(ADMISSION_ACTIONS):
            raise ValueError("base_costs must contain the complete action matrix")
        if any(not isinstance(row, tuple) or len(row) != 2 for row in self.base_costs):
            raise ValueError("base_costs must contain action/cost-row pairs")
        actions = tuple(_axis(row[0], ADMISSION_ACTIONS, "admission action") for row in self.base_costs)
        if actions != ADMISSION_ACTIONS:
            raise ValueError("base_costs must use canonical action order")
        rows: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        for action, row in self.base_costs:
            if not isinstance(row, tuple) or len(row) != len(SEMANTIC_STATUSES):
                raise ValueError("each base-cost row must cover the semantic axis")
            if any(not isinstance(item, tuple) or len(item) != 2 for item in row):
                raise ValueError("each base-cost row must contain status/cost pairs")
            statuses = tuple(_axis(item[0], SEMANTIC_STATUSES, "semantic status") for item in row)
            if statuses != SEMANTIC_STATUSES:
                raise ValueError("base_costs must use canonical semantic-status order")
            costs = tuple((status, _finite(cost, f"base_cost[{action}][{status}]")) for status, cost in row)
            rows.append((action, costs))
        object.__setattr__(self, "base_costs", tuple(rows))
        if not isinstance(self.accept_provenance_statuses, tuple):
            raise ValueError("accept_provenance_statuses must be a tuple")
        accepted_provenance = tuple(
            _axis(value, PROVENANCE_STATUSES, "provenance_status")
            for value in self.accept_provenance_statuses
        )
        if accepted_provenance != ("complete",):
            raise ValueError("accept_provenance_statuses must be exactly ('complete',)")
        if not isinstance(self.accept_auditability_statuses, tuple):
            raise ValueError("accept_auditability_statuses must be a tuple")
        accepted_auditability = tuple(
            _axis(value, AUDITABILITY_STATUSES, "evidence_auditability")
            for value in self.accept_auditability_statuses
        )
        if accepted_auditability != ("auditable",):
            raise ValueError("accept_auditability_statuses must be exactly ('auditable',)")
        if not isinstance(self.allow_accept_when_human_review_required, bool):
            raise ValueError("allow_accept_when_human_review_required must be boolean")
        payload = self.payload()
        object.__setattr__(self, "profile_hash", _derived_or_verified(self.profile_hash, "governance_profile", payload, "profile_hash"))

    def payload(self) -> dict[str, object]:
        return {"profile_id": self.profile_id, "base_costs": self.base_costs, "accept_provenance_statuses": self.accept_provenance_statuses, "accept_auditability_statuses": self.accept_auditability_statuses, "allow_accept_when_human_review_required": self.allow_accept_when_human_review_required}

    def base_cost(self, action: AdmissionAction, status: SemanticStatus) -> float:
        action = _axis(action, ADMISSION_ACTIONS, "admission action")
        status = _axis(status, SEMANTIC_STATUSES, "semantic status")
        return self.base_costs[ADMISSION_ACTIONS.index(action)][1][SEMANTIC_STATUSES.index(status)][1]


@dataclass(frozen=True)
class ActionEligibility:
    action: AdmissionAction
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _axis(self.action, ADMISSION_ACTIONS, "admission action"))
        if not isinstance(self.eligible, bool) or not isinstance(self.reason_codes, tuple):
            raise ValueError("eligibility must be boolean with tuple reason codes")
        reason_codes = tuple(
            _axis(code, ELIGIBILITY_REASON_CODES, "eligibility reason code")
            for code in self.reason_codes
        )
        object.__setattr__(self, "reason_codes", reason_codes)
        ordered = tuple(code for code in ELIGIBILITY_REASON_CODES if code in self.reason_codes)
        if ordered != self.reason_codes or len(set(self.reason_codes)) != len(self.reason_codes):
            raise ValueError("reason codes must be unique and in frozen order")
        allowed_reasons = {
            "accept": ELIGIBILITY_REASON_CODES[:3],
            "revise": ("revise_repair_not_verified",),
            "reject": (),
            "defer": (),
        }[self.action]
        if any(code not in allowed_reasons for code in self.reason_codes):
            raise ValueError("reason code is not applicable to its action")
        if self.action in ("reject", "defer"):
            if not self.eligible or self.reason_codes:
                raise ValueError("reject and defer must always be eligible without reasons")
        elif self.eligible != (not self.reason_codes):
            raise ValueError("eligible actions require no reasons and ineligible actions require reasons")


@dataclass(frozen=True)
class RoutingDecision:
    posterior_hash: str
    context_hash: str
    profile_hash: str
    router_id: str
    eligible_actions: tuple[AdmissionAction, ...]
    eligibility: tuple[ActionEligibility, ...]
    base_expected_risk_by_action: tuple[tuple[AdmissionAction, float], ...]
    chosen_action: AdmissionAction
    chosen_action_eligible: bool
    decision_hash: str = ""

    def __post_init__(self) -> None:
        for name in ("posterior_hash", "context_hash", "profile_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name), name))
        object.__setattr__(self, "router_id", _identifier(self.router_id, "router_id"))
        if not isinstance(self.eligible_actions, tuple):
            raise ValueError("eligible_actions must be a tuple")
        eligible_actions = tuple(
            _axis(action, ADMISSION_ACTIONS, "admission action")
            for action in self.eligible_actions
        )
        object.__setattr__(self, "eligible_actions", eligible_actions)
        canonical_actions = tuple(action for action in ADMISSION_ACTIONS if action in self.eligible_actions)
        if canonical_actions != self.eligible_actions:
            raise ValueError("eligible_actions must be unique and in canonical order")
        if not isinstance(self.eligibility, tuple) or any(not isinstance(item, ActionEligibility) for item in self.eligibility):
            raise ValueError("eligibility must be ActionEligibility values")
        if tuple(item.action for item in self.eligibility) != ADMISSION_ACTIONS:
            raise ValueError("eligibility must cover the action axis in canonical order")
        if self.eligible_actions != tuple(item.action for item in self.eligibility if item.eligible):
            raise ValueError("eligible_actions must agree with eligibility")
        if not isinstance(self.base_expected_risk_by_action, tuple):
            raise ValueError("base_expected_risk_by_action must be a tuple")
        if len(self.base_expected_risk_by_action) != len(ADMISSION_ACTIONS) or any(not isinstance(item, tuple) or len(item) != 2 for item in self.base_expected_risk_by_action):
            raise ValueError("base expected risks must contain action/risk pairs")
        risks = tuple(
            (
                _axis(action, ADMISSION_ACTIONS, "admission action"),
                _finite(risk, f"risk[{action}]"),
            )
            for action, risk in self.base_expected_risk_by_action
        )
        if tuple(action for action, _ in risks) != ADMISSION_ACTIONS:
            raise ValueError("base expected risks must cover the action axis in canonical order")
        object.__setattr__(self, "base_expected_risk_by_action", risks)
        object.__setattr__(self, "chosen_action", _axis(self.chosen_action, ADMISSION_ACTIONS, "admission action"))
        if not isinstance(self.chosen_action_eligible, bool):
            raise ValueError("chosen_action_eligible must be boolean")
        if self.chosen_action_eligible != next(item.eligible for item in self.eligibility if item.action == self.chosen_action):
            raise ValueError("chosen_action_eligible must agree with eligibility")
        payload = self.payload()
        object.__setattr__(self, "decision_hash", _derived_or_verified(self.decision_hash, "routing_decision", payload, "decision_hash"))

    def payload(self) -> dict[str, object]:
        return {"posterior_hash": self.posterior_hash, "context_hash": self.context_hash, "profile_hash": self.profile_hash, "router_id": self.router_id, "eligible_actions": self.eligible_actions, "eligibility": tuple({"action": item.action, "eligible": item.eligible, "reason_codes": item.reason_codes} for item in self.eligibility), "base_expected_risk_by_action": self.base_expected_risk_by_action, "chosen_action": self.chosen_action, "chosen_action_eligible": self.chosen_action_eligible}


def _audit_seed(estimator_id: str, seed: int | None) -> int | None:
    estimator_id = _identifier(estimator_id, "estimator_id")
    if seed is not None and (type(seed) is not int or seed not in (17, 29, 43)):
        raise ValueError("audit seed must be None, 17, 29, or 43")
    required_seed = {
        "semantic_majority_v1": None,
        "semantic_dawid_skene_v1": None,
        "semantic_softmax_v1": "softmax",
    }.get(estimator_id, "unknown")
    if required_seed == "unknown":
        raise ValueError("unknown estimator_id has no audit seed convention")
    if required_seed is None and seed is not None:
        raise ValueError("deterministic estimator audit records require seed=None")
    if required_seed == "softmax" and seed is None:
        raise ValueError("semantic_softmax_v1 audit records require seed 17, 29, or 43")
    return seed


@dataclass(frozen=True)
class PosteriorAuditRecord:
    audit_join_id: str
    fold_id: str
    seed: int | None
    posterior: SemanticPosterior
    record_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_join_id", _identifier(self.audit_join_id, "audit_join_id"))
        object.__setattr__(self, "fold_id", _identifier(self.fold_id, "fold_id"))
        if not isinstance(self.posterior, SemanticPosterior):
            raise ValueError("posterior must be SemanticPosterior")
        object.__setattr__(self, "seed", _audit_seed(self.posterior.estimator_id, self.seed))
        object.__setattr__(self, "record_hash", _derived_or_verified(self.record_hash, "posterior_audit_record", self.payload(), "record_hash"))

    def payload(self) -> dict[str, object]:
        return {"audit_join_id": self.audit_join_id, "fold_id": self.fold_id, "seed": self.seed, "posterior_hash": self.posterior.posterior_hash}


@dataclass(frozen=True)
class RouteAuditRecord:
    audit_join_id: str
    fold_id: str
    seed: int | None
    estimator_id: str
    posterior: SemanticPosterior
    decision: RoutingDecision
    record_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "audit_join_id", _identifier(self.audit_join_id, "audit_join_id"))
        object.__setattr__(self, "fold_id", _identifier(self.fold_id, "fold_id"))
        object.__setattr__(self, "estimator_id", _identifier(self.estimator_id, "estimator_id"))
        if not isinstance(self.posterior, SemanticPosterior):
            raise ValueError("posterior must be SemanticPosterior")
        if not isinstance(self.decision, RoutingDecision):
            raise ValueError("decision must be RoutingDecision")
        if self.estimator_id != self.posterior.estimator_id:
            raise ValueError("estimator_id must be derived from the authoritative posterior")
        if self.decision.posterior_hash != self.posterior.posterior_hash:
            raise ValueError("decision posterior hash does not resolve to authoritative posterior")
        object.__setattr__(self, "seed", _audit_seed(self.estimator_id, self.seed))
        object.__setattr__(self, "record_hash", _derived_or_verified(self.record_hash, "route_audit_record", self.payload(), "record_hash"))

    def payload(self) -> dict[str, object]:
        """The serialized route-audit envelope; posterior linkage is independently verified."""
        return {"audit_join_id": self.audit_join_id, "fold_id": self.fold_id, "seed": self.seed, "estimator_id": self.estimator_id, "decision_hash": self.decision.decision_hash}


def semantic_observations_from_mapping(value: Mapping[str, object]) -> SemanticObservations:
    value = _exact_mapping(value, ("content_support_score", "contradiction_score", "partial_support_score", "uncertainty_score"), "observations")
    return SemanticObservations(**value)


def semantic_vote_from_mapping(value: Mapping[str, object]) -> SemanticVote:
    value = _exact_mapping(value, ("validator_id", "status", "confidence"), "semantic vote")
    return SemanticVote(**value)


def semantic_input_from_mapping(value: Mapping[str, object]) -> SemanticEstimatorInput:
    value = _exact_mapping(value, ("observations", "semantic_votes"), "semantic input")
    votes = value["semantic_votes"]
    if not isinstance(votes, list):
        raise ValueError("semantic_votes mapping value must be a list")
    return SemanticEstimatorInput(semantic_observations_from_mapping(value["observations"]), tuple(semantic_vote_from_mapping(vote) for vote in votes))


def semantic_posterior(*, estimator_id: str, estimator_state_hash: str, semantic_input: SemanticEstimatorInput, probabilities: Mapping[str, object]) -> SemanticPosterior:
    if not isinstance(semantic_input, SemanticEstimatorInput):
        raise ValueError("semantic_input must be SemanticEstimatorInput")
    probabilities = _exact_mapping(probabilities, SEMANTIC_STATUSES, "probabilities")
    return SemanticPosterior(estimator_id, estimator_state_hash, semantic_input.semantic_input_hash, tuple((status, probabilities[status]) for status in SEMANTIC_STATUSES))


def governance_context(provenance_status: str, evidence_auditability: str, human_review_required: bool, repairability: str, policy_profile_id: str) -> GovernanceContext:
    return GovernanceContext(provenance_status, evidence_auditability, human_review_required, repairability, policy_profile_id)


def governance_context_from_mapping(value: Mapping[str, object]) -> GovernanceContext:
    value = _exact_mapping(value, ("provenance_status", "evidence_auditability", "human_review_required", "repairability", "policy_profile_id"), "governance context")
    return GovernanceContext(**value)


def governance_profile(*, profile_id: str, base_costs: Mapping[str, Mapping[str, object]], accept_provenance_statuses: tuple[str, ...] = ("complete",), accept_auditability_statuses: tuple[str, ...] = ("auditable",), allow_accept_when_human_review_required: bool = False) -> GovernanceProfile:
    base_costs = _exact_mapping(base_costs, ADMISSION_ACTIONS, "base_costs")
    rows = []
    for action in ADMISSION_ACTIONS:
        row = _exact_mapping(base_costs[action], SEMANTIC_STATUSES, f"base_costs[{action}]")
        rows.append((action, tuple((status, row[status]) for status in SEMANTIC_STATUSES)))
    return GovernanceProfile(profile_id, tuple(rows), accept_provenance_statuses, accept_auditability_statuses, allow_accept_when_human_review_required)


def routing_decision(*, posterior_hash: str, context_hash: str, profile_hash: str, router_id: str, eligible_actions: tuple[AdmissionAction, ...], eligibility: tuple[ActionEligibility, ...], base_expected_risk_by_action: tuple[tuple[AdmissionAction, float], ...], chosen_action: AdmissionAction, chosen_action_eligible: bool) -> RoutingDecision:
    return RoutingDecision(posterior_hash, context_hash, profile_hash, router_id, eligible_actions, eligibility, base_expected_risk_by_action, chosen_action, chosen_action_eligible)


def posterior_audit_record(audit_join_id: str, fold_id: str, seed: int | None, posterior: SemanticPosterior) -> PosteriorAuditRecord:
    return PosteriorAuditRecord(audit_join_id, fold_id, seed, posterior)


def route_audit_record(audit_join_id: str, fold_id: str, seed: int | None, posterior: SemanticPosterior, decision: RoutingDecision) -> RouteAuditRecord:
    if not isinstance(posterior, SemanticPosterior) or not isinstance(decision, RoutingDecision):
        raise ValueError("posterior and decision must be r12 value objects")
    if decision.posterior_hash != posterior.posterior_hash:
        raise ValueError("decision posterior hash does not resolve to supplied posterior")
    return RouteAuditRecord(audit_join_id, fold_id, seed, posterior.estimator_id, posterior, decision)


def migrate_legacy_semantic_status(label: str) -> SemanticStatus:
    mapping = {"supported": "supported", "contradicted": "contradicted", "repairable": "partially_supported", "insufficient": "insufficient_evidence"}
    label = _identifier(label, "legacy semantic label")
    if label not in mapping:
        raise ValueError("unsupported legacy semantic label")
    return mapping[label]


__all__ = [
    "ADMISSION_ACTIONS", "AUDITABILITY_STATUSES", "ActionEligibility", "AdmissionAction",
    "DOMAIN_VERSION", "ELIGIBILITY_REASON_CODES", "GovernanceContext", "GovernanceProfile",
    "PosteriorAuditRecord", "PROVENANCE_STATUSES", "REPAIRABILITY_STATUSES", "RouteAuditRecord",
    "RoutingDecision", "SEMANTIC_STATUSES", "SemanticEstimatorInput", "SemanticObservations",
    "SemanticPosterior", "SemanticStatus", "SemanticVote", "canonical_json_bytes",
    "governance_context", "governance_context_from_mapping", "governance_profile",
    "migrate_legacy_semantic_status", "posterior_audit_record", "route_audit_record",
    "routing_decision", "semantic_input_from_mapping", "semantic_observations_from_mapping",
    "semantic_posterior", "semantic_vote_from_mapping",
]
