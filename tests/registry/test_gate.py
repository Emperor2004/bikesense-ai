"""Unit tests for promotion gate decision logic and thresholds."""



def check_gate_decision(
    champ_mae: float,
    chall_mae: float,
    champ_shortfall: float,
    chall_shortfall: float,
    margin_pct: float = 3.0,
    max_shortfall_increase_pct: float = 5.0,
) -> tuple[bool, str]:
    """Testable gate evaluation logic."""
    mae_diff = champ_mae - chall_mae
    mae_pct_improvement = (mae_diff / champ_mae) * 100.0

    shortfall_diff = chall_shortfall - champ_shortfall
    shortfall_pct_increase = (
        (shortfall_diff / champ_shortfall) * 100.0 if champ_shortfall > 0 else 0.0
    )

    if mae_pct_improvement < margin_pct:
        return False, f"MAE improvement {mae_pct_improvement:.2f}% < {margin_pct}%"

    if shortfall_pct_increase > max_shortfall_increase_pct:
        return False, f"Shortfall increase {shortfall_pct_increase:.2f}% > {max_shortfall_increase_pct}%"

    return True, "Passed"


def test_gate_rejection_below_margin():
    # 2% improvement when 3% is required
    passed, reason = check_gate_decision(
        champ_mae=10.0,
        chall_mae=9.8,
        champ_shortfall=100.0,
        chall_shortfall=98.0,
        margin_pct=3.0,
    )
    assert not passed
    assert "MAE improvement" in reason


def test_gate_approval_above_margin():
    # 5% improvement
    passed, reason = check_gate_decision(
        champ_mae=10.0,
        chall_mae=9.5,
        champ_shortfall=100.0,
        chall_shortfall=95.0,
        margin_pct=3.0,
    )
    assert passed
    assert reason == "Passed"


def test_gate_shortfall_guardrail_rejection():
    # 4% MAE improvement, but shortfall increases by 8% (fails guardrail)
    passed, reason = check_gate_decision(
        champ_mae=10.0,
        chall_mae=9.6,
        champ_shortfall=100.0,
        chall_shortfall=108.0,
        margin_pct=3.0,
        max_shortfall_increase_pct=5.0,
    )
    assert not passed
    assert "Shortfall increase" in reason
