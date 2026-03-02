"""Built-in EXOKERN baseline results for comparison."""

BASELINES = {
    "PegInsert": {
        "env": "Isaac-Forge-PegInsert-Direct-v0",
        "version": "v0",
        "full_ft": {
            "success_rate": 100.0,
            "avg_force": 3.2,
            "peak_force": 10.8,
            "n_episodes": 300,
            "seeds": [42, 123, 7],
            "source": "EXOKERN/skill-forge-peginsert-v0",
        },
        "no_ft": {
            "success_rate": 100.0,
            "avg_force": 5.2,
            "peak_force": 12.2,
            "n_episodes": 300,
            "seeds": [42, 123, 7],
            "source": "EXOKERN/skill-forge-peginsert-v0",
        },
    },
}


def get_baseline(env_name: str, condition: str = "full_ft") -> dict | None:
    """Look up the EXOKERN baseline for a given environment."""
    for key, entry in BASELINES.items():
        if entry["env"] == env_name or key.lower() in env_name.lower():
            return entry.get(condition)
    return None
