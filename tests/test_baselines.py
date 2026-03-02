from exokern_eval.baselines import get_baseline


def test_peginsert_baseline_exists():
    bl = get_baseline("Isaac-Forge-PegInsert-Direct-v0", "full_ft")
    assert bl is not None
    assert bl["success_rate"] == 100.0
    assert bl["avg_force"] == 3.2


def test_unknown_env_returns_none():
    assert get_baseline("NonExistent-Env-v99") is None
