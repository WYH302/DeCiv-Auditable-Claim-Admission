"""Pure r12 governance eligibility and admission routing."""

from __future__ import annotations

from .contracts import (
    ADMISSION_ACTIONS,
    ELIGIBILITY_REASON_CODES,
    SEMANTIC_STATUSES,
    AdmissionAction,
    ActionEligibility,
    GovernanceContext,
    GovernanceProfile,
    RoutingDecision,
    SemanticPosterior,
    routing_decision,
)


DIRECT_MAPPING_ROUTER_ID = "direct_mapping_v1"
MINIMUM_CONTEXTUAL_RISK_ROUTER_ID = "minimum_contextual_risk_v1"
_MISSING = object()


def _validate_inputs(
    context: GovernanceContext,
    profile: GovernanceProfile,
    posterior: object = _MISSING,
) -> None:
    if not isinstance(context, GovernanceContext):
        raise ValueError("context must be GovernanceContext")
    if not isinstance(profile, GovernanceProfile):
        raise ValueError("profile must be GovernanceProfile")
    if posterior is not _MISSING and not isinstance(posterior, SemanticPosterior):
        raise ValueError("posterior must be SemanticPosterior")
    if context.policy_profile_id != profile.profile_id:
        raise ValueError("context policy profile does not match profile")


def _eligibility_trace(
    context: GovernanceContext,
    profile: GovernanceProfile,
) -> tuple[ActionEligibility, ...]:
    accept_reasons = tuple(
        reason
        for reason, applies in (
            (
                ELIGIBILITY_REASON_CODES[0],
                context.provenance_status not in profile.accept_provenance_statuses,
            ),
            (
                ELIGIBILITY_REASON_CODES[1],
                context.evidence_auditability not in profile.accept_auditability_statuses,
            ),
            (
                ELIGIBILITY_REASON_CODES[2],
                context.human_review_required
                and not profile.allow_accept_when_human_review_required,
            ),
        )
        if applies
    )
    revise_reasons = (
        ()
        if context.repairability == "verified_repair"
        else (ELIGIBILITY_REASON_CODES[3],)
    )
    return (
        ActionEligibility("accept", not accept_reasons, accept_reasons),
        ActionEligibility("reject", True, ()),
        ActionEligibility("revise", not revise_reasons, revise_reasons),
        ActionEligibility("defer", True, ()),
    )


def _base_expected_risks(
    posterior: SemanticPosterior,
    profile: GovernanceProfile,
) -> tuple[tuple[AdmissionAction, float], ...]:
    return tuple(
        (
            action,
            sum(
                (
                    posterior.probability(status) * profile.base_cost(action, status)
                    for status in SEMANTIC_STATUSES
                ),
                0.0,
            ),
        )
        for action in ADMISSION_ACTIONS
    )


def _direct_action(posterior: SemanticPosterior) -> AdmissionAction:
    status = max(SEMANTIC_STATUSES, key=posterior.probability)
    return ADMISSION_ACTIONS[SEMANTIC_STATUSES.index(status)]


def eligible_actions(
    context: GovernanceContext,
    profile: GovernanceProfile,
) -> tuple[AdmissionAction, ...]:
    """Return the canonical nonempty admission-action set permitted by policy."""
    _validate_inputs(context, profile)
    trace = _eligibility_trace(context, profile)
    return tuple(item.action for item in trace if item.eligible)


def route_direct_mapping(
    posterior: SemanticPosterior,
    context: GovernanceContext,
    profile: GovernanceProfile,
) -> RoutingDecision:
    """Map the semantic argmax positionally without repairing ineligible actions."""
    _validate_inputs(context, profile, posterior)
    trace = _eligibility_trace(context, profile)
    actions = tuple(item.action for item in trace if item.eligible)
    chosen = _direct_action(posterior)
    return routing_decision(
        posterior_hash=posterior.posterior_hash,
        context_hash=context.context_hash,
        profile_hash=profile.profile_hash,
        router_id=DIRECT_MAPPING_ROUTER_ID,
        eligible_actions=actions,
        eligibility=trace,
        base_expected_risk_by_action=_base_expected_risks(posterior, profile),
        chosen_action=chosen,
        chosen_action_eligible=chosen in actions,
    )


def route_minimum_contextual_risk(
    posterior: SemanticPosterior,
    context: GovernanceContext,
    profile: GovernanceProfile,
) -> RoutingDecision:
    """Choose the canonical eligible action with minimum raw base expected risk."""
    _validate_inputs(context, profile, posterior)
    trace = _eligibility_trace(context, profile)
    actions = tuple(item.action for item in trace if item.eligible)
    risks = _base_expected_risks(posterior, profile)
    chosen = min(
        actions,
        key=lambda action: risks[ADMISSION_ACTIONS.index(action)][1],
    )
    return routing_decision(
        posterior_hash=posterior.posterior_hash,
        context_hash=context.context_hash,
        profile_hash=profile.profile_hash,
        router_id=MINIMUM_CONTEXTUAL_RISK_ROUTER_ID,
        eligible_actions=actions,
        eligibility=trace,
        base_expected_risk_by_action=risks,
        chosen_action=chosen,
        chosen_action_eligible=True,
    )


__all__ = [
    "DIRECT_MAPPING_ROUTER_ID",
    "MINIMUM_CONTEXTUAL_RISK_ROUTER_ID",
    "eligible_actions",
    "route_direct_mapping",
    "route_minimum_contextual_risk",
]
