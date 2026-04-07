"""
Greenhouse Climate Control — Structured Test Case Definitions.

Each test case is a dict with:
    id, description, initial_state, actions, expected, edge_case
"""

# ─── Observation Field Sets ────────────────────────────────────────────────
REQUIRED_OBS_FIELDS = [
    "temperature", "humidity", "co2_level", "light_intensity",
    "outside_temperature", "outside_humidity", "cloud_cover",
    "hour_of_day", "day_number",
    "plant_health", "growth_progress",
    "energy_consumed_step", "total_energy_consumed",
    "energy_cost_step", "total_energy_cost",
    "step_number", "max_steps", "task_id",
    "status_message",
    "done", "reward",
]

REQUIRED_STATE_FIELDS = [
    "episode_id", "step_count",
]


# ─── Test Cases ────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── 1. Reset produces valid initial observation ─────────────────────
    {
        "id": "TC-001",
        "description": "Reset returns valid initial observation with all fields",
        "category": "reset",
        "actions": [],  # just reset
        "checks": [
            ("obs.done", "==", False),
            ("obs.reward", "==", 0.0),
            ("obs.step_number", "==", 0),
            ("obs.plant_health", "==", 1.0),
            ("obs.growth_progress", "==", 0.0),
            ("obs.total_energy_consumed", "==", 0.0),
            ("obs.total_energy_cost", "==", 0.0),
            ("obs.task_id", "in", ["maintain_temperature", "optimize_growth", "weather_resilience"]),
        ],
        "edge_case": False,
    },

    # ── 2. Reset idempotency — double reset ────────────────────────────
    {
        "id": "TC-002",
        "description": "Double reset clears state cleanly",
        "category": "reset",
        "actions": [
            {"heater_power": 1.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
            "RESET",
        ],
        "checks": [
            ("obs.step_number", "==", 0),
            ("obs.plant_health", "==", 1.0),
            ("obs.growth_progress", "==", 0.0),
            ("obs.total_energy_consumed", "==", 0.0),
            ("obs.done", "==", False),
        ],
        "edge_case": False,
    },

    # ── 3. Zero action — all actuators off ─────────────────────────────
    {
        "id": "TC-003",
        "description": "Zero action (all off) consumes no energy and advances step",
        "category": "normal",
        "actions": [
            {"heater_power": 0.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
        ],
        "checks": [
            ("obs.step_number", "==", 1),
            ("obs.energy_consumed_step", "==", 0.0),
            ("obs.energy_cost_step", "==", 0.0),
            ("obs.done", "==", False),
            ("obs.reward", ">=", 0.0),
            ("obs.reward", "<=", 1.0),
        ],
        "edge_case": False,
    },

    # ── 4. Full-power action — energy is consumed ──────────────────────
    {
        "id": "TC-004",
        "description": "Full power on all actuators consumes maximum energy",
        "category": "normal",
        "actions": [
            {"heater_power": 1.0, "ventilation_rate": 1.0, "humidifier_level": 1.0, "artificial_lighting": 1.0},
        ],
        "checks": [
            ("obs.step_number", "==", 1),
            ("obs.energy_consumed_step", ">", 0.0),
            ("obs.total_energy_consumed", ">", 0.0),
            ("obs.energy_cost_step", ">", 0.0),
            ("obs.done", "==", False),
            ("obs.reward", ">=", 0.0),
            ("obs.reward", "<=", 1.0),
        ],
        "edge_case": False,
    },

    # ── 5. Energy calculation correctness ──────────────────────────────
    {
        "id": "TC-005",
        "description": "Energy = heater*2.0 + vent*0.3 + humid*0.2 + light*1.5 kWh",
        "category": "normal",
        "actions": [
            {"heater_power": 1.0, "ventilation_rate": 1.0, "humidifier_level": 1.0, "artificial_lighting": 1.0},
        ],
        "checks": [
            # total should be 2.0+0.3+0.2+1.5 = 4.0 kWh
            ("obs.energy_consumed_step", "approx", 4.0),
            # cost = 4.0 * 0.15 = 0.60
            ("obs.energy_cost_step", "approx", 0.60),
        ],
        "edge_case": False,
    },

    # ── 6. Step count increments correctly ─────────────────────────────
    {
        "id": "TC-006",
        "description": "Step count increments by 1 on each step",
        "category": "normal",
        "actions": [
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.3, "artificial_lighting": 0.0},
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.3, "artificial_lighting": 0.0},
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.3, "artificial_lighting": 0.0},
        ],
        "checks": [
            ("obs.step_number", "==", 3),
        ],
        "edge_case": False,
    },

    # ── 7. Reward always in [0.0, 1.0] ─────────────────────────────────
    {
        "id": "TC-007",
        "description": "Reward is bounded in [0.0, 1.0] for any action",
        "category": "boundary",
        "actions": [
            {"heater_power": 1.0, "ventilation_rate": 1.0, "humidifier_level": 1.0, "artificial_lighting": 1.0},
            {"heater_power": 0.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
            {"heater_power": 0.5, "ventilation_rate": 0.5, "humidifier_level": 0.5, "artificial_lighting": 0.5},
        ],
        "checks": [
            # We check ALL steps (handled by multi-step validator)
            ("all_rewards", ">=", 0.0),
            ("all_rewards", "<=", 1.0),
        ],
        "edge_case": False,
    },

    # ── 8. Plant health starts at 1.0 and changes ─────────────────────
    {
        "id": "TC-008",
        "description": "Plant health stays near 1.0 under good conditions",
        "category": "normal",
        "actions": [
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.3, "artificial_lighting": 0.5},
        ] * 5,
        "checks": [
            ("obs.plant_health", ">=", 0.8),
            ("obs.plant_health", "<=", 1.0),
        ],
        "edge_case": False,
    },

    # ── 9. Time advances: hour_of_day increments each step ─────────────
    {
        "id": "TC-009",
        "description": "Time advances by ~1 hour per step",
        "category": "normal",
        "actions": [
            {"heater_power": 0.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
        ],
        "checks": [
            # After 1 step from hour 8.0, should be 9.0
            ("obs.hour_of_day", ">=", 8.0),
            ("obs.hour_of_day", "<=", 10.0),
        ],
        "edge_case": False,
    },

    # ── 10. Temperature responds to heater ─────────────────────────────
    {
        "id": "TC-010",
        "description": "Heater raises indoor temperature above initial",
        "category": "normal",
        "actions": [
            {"heater_power": 1.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
            {"heater_power": 1.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
            {"heater_power": 1.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
        ],
        "checks": [
            # After 3 full-power heater steps, temp should be above typical start (~17°C)
            ("obs.temperature", ">", 18.0),
        ],
        "edge_case": False,
    },

    # ── 11. Episode terminates at max_steps ────────────────────────────
    {
        "id": "TC-011",
        "description": "Episode ends at max_steps (24 for maintain_temperature)",
        "category": "boundary",
        "actions": [  # 24 zero-actions
            {"heater_power": 0.3, "ventilation_rate": 0.1, "humidifier_level": 0.2, "artificial_lighting": 0.0},
        ] * 24,
        "checks": [
            ("obs.done", "==", True),
            ("obs.step_number", "==", 24),
        ],
        "edge_case": False,
    },

    # ── 12. Grader score appears in metadata on done ───────────────────
    {
        "id": "TC-012",
        "description": "grader_score is in metadata when done=True",
        "category": "boundary",
        "actions": [
            {"heater_power": 0.3, "ventilation_rate": 0.1, "humidifier_level": 0.2, "artificial_lighting": 0.0},
        ] * 24,
        "checks": [
            ("obs.done", "==", True),
            ("metadata.grader_score", ">=", 0.0),
            ("metadata.grader_score", "<=", 1.0),
        ],
        "edge_case": False,
    },

    # ── 13. State endpoint matches observation ─────────────────────────
    {
        "id": "TC-013",
        "description": "GET /state step_count matches observation step_number",
        "category": "state",
        "actions": [
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.3, "artificial_lighting": 0.0},
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.3, "artificial_lighting": 0.0},
        ],
        "checks": [
            ("state.step_count", "==", 2),
        ],
        "edge_case": False,
    },

    # ── 14. Action echo in observation ─────────────────────────────────
    {
        "id": "TC-014",
        "description": "last_action in observation echoes the sent action",
        "category": "normal",
        "actions": [
            {"heater_power": 0.75, "ventilation_rate": 0.25, "humidifier_level": 0.50, "artificial_lighting": 0.10},
        ],
        "checks": [
            ("obs.last_action.heater_power", "approx", 0.75),
            ("obs.last_action.ventilation_rate", "approx", 0.25),
            ("obs.last_action.humidifier_level", "approx", 0.50),
            ("obs.last_action.artificial_lighting", "approx", 0.10),
        ],
        "edge_case": False,
    },

    # ── 15. Growth progress increases under good conditions ────────────
    {
        "id": "TC-015",
        "description": "Growth progress > 0 after several steps with good actions",
        "category": "normal",
        "actions": [
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.4, "artificial_lighting": 0.6},
        ] * 10,
        "checks": [
            ("obs.growth_progress", ">", 0.0),
        ],
        "edge_case": False,
    },

    # ── 16. Cumulative energy grows monotonically ──────────────────────
    {
        "id": "TC-016",
        "description": "total_energy_consumed never decreases across steps",
        "category": "boundary",
        "actions": [
            {"heater_power": 0.5, "ventilation_rate": 0.2, "humidifier_level": 0.3, "artificial_lighting": 0.4},
            {"heater_power": 0.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
            {"heater_power": 1.0, "ventilation_rate": 1.0, "humidifier_level": 1.0, "artificial_lighting": 1.0},
        ],
        "checks": [
            ("energy_monotonic", "==", True),
        ],
        "edge_case": False,
    },

    # ── 17. Temperature in physical bounds ─────────────────────────────
    {
        "id": "TC-017",
        "description": "Temperature always in [-10, 50] regardless of actions",
        "category": "boundary",
        "actions": [
            {"heater_power": 1.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
        ] * 20,
        "checks": [
            ("all_temperatures", ">=", -10.0),
            ("all_temperatures", "<=", 50.0),
        ],
        "edge_case": True,
    },

    # ── 18. Humidity in physical bounds ─────────────────────────────────
    {
        "id": "TC-018",
        "description": "Humidity always in [5, 100] regardless of actions",
        "category": "boundary",
        "actions": [
            {"heater_power": 0.0, "ventilation_rate": 0.0, "humidifier_level": 1.0, "artificial_lighting": 0.0},
        ] * 20,
        "checks": [
            ("all_humidities", ">=", 5.0),
            ("all_humidities", "<=", 100.0),
        ],
        "edge_case": True,
    },

    # ── 19. CO₂ in physical bounds ─────────────────────────────────────
    {
        "id": "TC-019",
        "description": "CO₂ always in [200, 3000] regardless of actions",
        "category": "boundary",
        "actions": [
            {"heater_power": 0.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
        ] * 20,
        "checks": [
            ("all_co2", ">=", 200.0),
            ("all_co2", "<=", 3000.0),
        ],
        "edge_case": True,
    },

    # ── 20. Default action (fields omitted) fills zeros ────────────────
    {
        "id": "TC-020",
        "description": "Omitted action fields default to 0.0 (no crash)",
        "category": "edge",
        "actions": [
            {},  # empty action → all defaults
        ],
        "checks": [
            ("obs.step_number", "==", 1),
            ("obs.done", "==", False),
        ],
        "edge_case": True,
    },

    # ── 21. Status message is non-empty string after step ──────────────
    {
        "id": "TC-021",
        "description": "status_message is a non-empty human-readable string",
        "category": "normal",
        "actions": [
            {"heater_power": 0.5, "ventilation_rate": 0.1, "humidifier_level": 0.3, "artificial_lighting": 0.0},
        ],
        "checks": [
            ("obs.status_message", "len>", 10),
        ],
        "edge_case": False,
    },

    # ── 22. max_steps matches task (24 for easy) ───────────────────────
    {
        "id": "TC-022",
        "description": "max_steps=24 for maintain_temperature task",
        "category": "normal",
        "actions": [],
        "checks": [
            ("obs.max_steps", "==", 24),
        ],
        "edge_case": False,
    },

    # ── 23. Cloud cover in [0, 1] ──────────────────────────────────────
    {
        "id": "TC-023",
        "description": "Cloud cover is always within [0.0, 1.0]",
        "category": "boundary",
        "actions": [
            {"heater_power": 0.0, "ventilation_rate": 0.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
        ] * 10,
        "checks": [
            ("all_cloud_covers", ">=", 0.0),
            ("all_cloud_covers", "<=", 1.0),
        ],
        "edge_case": False,
    },

    # ── 24. Plant health in [0, 1] ─────────────────────────────────────
    {
        "id": "TC-024",
        "description": "Plant health always bounded [0.0, 1.0]",
        "category": "boundary",
        "actions": [
            {"heater_power": 0.0, "ventilation_rate": 1.0, "humidifier_level": 0.0, "artificial_lighting": 0.0},
        ] * 20,
        "checks": [
            ("all_health", ">=", 0.0),
            ("all_health", "<=", 1.0),
        ],
        "edge_case": True,
    },
]
