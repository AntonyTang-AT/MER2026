import pytest

from src.inference.stage_manager import StageManager


def test_submit_plan_includes_submission():
    mgr = StageManager()
    plan = mgr.build_plan(mode="submit")
    assert plan.routing_split == "candidate"
    assert plan.submission is True
    assert plan.evaluation is False


def test_eval_plan_uses_human_routing():
    mgr = StageManager()
    plan = mgr.build_plan(mode="eval")
    assert plan.routing_split == "human"
    assert plan.submission is False
    assert plan.evaluation is True


def test_routing_required_when_use_routing_in_prompt():
    mgr = StageManager(
        {
            "stages": {
                "routing": {"enabled": False},
                "affectgpt": {"enabled": True, "use_routing_in_prompt": True},
            }
        }
    )
    with pytest.raises(ValueError, match="routing"):
        mgr.build_plan(mode="submit")
