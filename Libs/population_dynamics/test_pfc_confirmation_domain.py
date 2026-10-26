"""Physical-domain completion preserves absence of numerical evidence."""
import json
import pytest

from .pfc_confirmation_domain import complete_domain, NEW_STATUS


def fixture():
    base = {"left_cohort": "coarse", "right_cohort": "fine", "condition": "cue", "phase": "delay", "window_ms": "10.0", "field_index": "0"}
    domains = [{"family": "count_cdf", "endpoint_identity": json.dumps(base | {"count_threshold": str(k)}, sort_keys=True),
                "empirical_status": "EMPIRICALLY_DEGENERATE", "definition_status": "EMPIRICALLY_DEGENERATE_UNCERTAIN",
                "reason": "no_population_uncertainty_claim", "physical_maximum_count": "5", "estimate": "0.0"} for k in range(2)]
    domains.append({"family": "field_lag_covariance", "endpoint_identity": json.dumps({"lag_ms": "100.0"}),
                    "empirical_status": "UNDEFINED_PHASE_CONTAINED_LAG", "definition_status": "UNDEFINED_NO_WITHIN_PHASE_PAIRS",
                    "reason": "lag_not_shorter_than_phase", "physical_maximum_count": "", "estimate": ""})
    support = [base | {"physical_maximum_count": "5", "archived_threshold_minimum": "0", "archived_threshold_maximum": "1",
                       "unrepresented_nontrivial_thresholds": "3", "first_unrepresented_threshold": "2"}]
    return domains, support


def test_missing_thresholds_have_no_imputed_estimate_or_structural_zero():
    domains, support = fixture();rows, report = complete_domain(domains, support)
    assert rows[:3] == domains
    assert [json.loads(row["endpoint_identity"])["count_threshold"] for row in rows[3:]] == ["2", "3", "4"]
    assert all(row["estimate"] == "" and row["definition_status"] == NEW_STATUS for row in rows[3:])
    assert report["definition_status_counts"]["UNDEFINED_NO_WITHIN_PHASE_PAIRS"] == 1
    assert report["confirmation_execution_released"] is False


def test_duplicate_endpoint_cannot_inflate_domain():
    domains, support = fixture()
    with pytest.raises(ValueError, match="DUPLICATE_ENDPOINT_IDENTITY"):
        complete_domain(domains + [domains[0]], support)


def test_source_missing_count_mismatch_is_rejected():
    domains, support = fixture();support[0]["unrepresented_nontrivial_thresholds"] = "2"
    with pytest.raises(ValueError, match="MISSING_THRESHOLD_COUNT_MISMATCH"):
        complete_domain(domains, support)


def test_refractory_support_bound_mismatch_is_rejected():
    domains, support = fixture();support[0]["physical_maximum_count"] = "6"
    with pytest.raises(ValueError, match="SUPPORT_BOUND_MISMATCH"):
        complete_domain(domains, support)
