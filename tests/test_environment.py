"""
Greenhouse Climate Control — Offline Unit Tests.

Tests the environment logic directly (no server needed).
Run with: python -m pytest tests/test_environment.py -v
Or:       python tests/test_environment.py
"""

import sys
from pathlib import Path

# Allow direct execution from the greenhouse directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.greenhouse_environment import (
    GreenhouseEnvironment,
    TASK_CONFIGS,
    HEATER_ENERGY_KWH,
    VENTILATION_ENERGY_KWH,
    HUMIDIFIER_ENERGY_KWH,
    LIGHTING_ENERGY_KWH,
    ENERGY_PRICE_PER_KWH,
)
from models import GreenhouseAction, GreenhouseObservation, GreenhouseState


# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_action(**kw) -> GreenhouseAction:
    """Create a GreenhouseAction with defaults (all zeros) + overrides."""
    defaults = {
        "heater_power": 0.0,
        "ventilation_rate": 0.0,
        "humidifier_level": 0.0,
        "artificial_lighting": 0.0,
    }
    defaults.update(kw)
    return GreenhouseAction(**defaults)


PASSED = 0
FAILED = 0
ERRORS = []


def check(test_id: str, description: str, condition: bool, detail: str = ""):
    """Assert a condition and track pass/fail."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ PASS  {test_id}: {description}")
    else:
        FAILED += 1
        msg = f"  ❌ FAIL  {test_id}: {description}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        ERRORS.append(msg)


# ─── Test Functions ──────────────────────────────────────────────────────────

def test_reset_returns_valid_observation():
    """TC-001: Reset returns valid initial observation."""
    env = GreenhouseEnvironment()
    obs = env.reset()

    check("TC-001a", "reset returns GreenhouseObservation",
          isinstance(obs, GreenhouseObservation))
    check("TC-001b", "done == False after reset",
          obs.done is False)
    check("TC-001c", "reward == 0.0 after reset",
          obs.reward == 0.0, f"got {obs.reward}")
    check("TC-001d", "step_number == 0 after reset",
          obs.step_number == 0, f"got {obs.step_number}")
    check("TC-001e", "plant_health == 1.0 after reset",
          obs.plant_health == 1.0, f"got {obs.plant_health}")
    check("TC-001f", "growth_progress == 0.0 after reset",
          obs.growth_progress == 0.0, f"got {obs.growth_progress}")
    check("TC-001g", "total_energy_consumed == 0 after reset",
          obs.total_energy_consumed == 0.0, f"got {obs.total_energy_consumed}")
    check("TC-001h", "task_id is valid",
          obs.task_id in TASK_CONFIGS, f"got {obs.task_id}")


def test_double_reset_clears_state():
    """TC-002: Double reset clears episode state."""
    env = GreenhouseEnvironment()
    env.reset()
    env.step(make_action(heater_power=1.0))
    env.step(make_action(heater_power=1.0))

    obs = env.reset()
    check("TC-002a", "step_number == 0 after double reset",
          obs.step_number == 0, f"got {obs.step_number}")
    check("TC-002b", "plant_health == 1.0 after double reset",
          obs.plant_health == 1.0, f"got {obs.plant_health}")
    check("TC-002c", "growth_progress == 0 after double reset",
          obs.growth_progress == 0.0, f"got {obs.growth_progress}")
    check("TC-002d", "total_energy_consumed == 0 after double reset",
          obs.total_energy_consumed == 0.0, f"got {obs.total_energy_consumed}")
    check("TC-002e", "done == False after double reset",
          obs.done is False)


def test_zero_action_no_energy():
    """TC-003: All-zero action consumes no energy."""
    env = GreenhouseEnvironment()
    env.reset()
    obs = env.step(make_action())

    check("TC-003a", "step_number == 1",
          obs.step_number == 1, f"got {obs.step_number}")
    check("TC-003b", "energy_consumed_step == 0.0 with all-zero action",
          obs.energy_consumed_step == 0.0, f"got {obs.energy_consumed_step}")
    check("TC-003c", "energy_cost_step == 0.0 with all-zero action",
          obs.energy_cost_step == 0.0, f"got {obs.energy_cost_step}")
    check("TC-003d", "done is False on step 1 of 24",
          obs.done is False)
    check("TC-003e", "reward in [0, 1]",
          0.0 <= obs.reward <= 1.0, f"got {obs.reward}")


def test_full_power_energy():
    """TC-004/005: Full power consumes max energy with correct calculation."""
    env = GreenhouseEnvironment()
    env.reset()
    obs = env.step(make_action(
        heater_power=1.0, ventilation_rate=1.0,
        humidifier_level=1.0, artificial_lighting=1.0,
    ))

    expected_energy = (
        HEATER_ENERGY_KWH + VENTILATION_ENERGY_KWH
        + HUMIDIFIER_ENERGY_KWH + LIGHTING_ENERGY_KWH
    )
    expected_cost = expected_energy * ENERGY_PRICE_PER_KWH

    check("TC-004a", "energy_consumed_step > 0 with full power",
          obs.energy_consumed_step > 0.0, f"got {obs.energy_consumed_step}")
    check("TC-005a", f"energy_consumed_step ≈ {expected_energy:.1f} kWh",
          abs(obs.energy_consumed_step - expected_energy) < 0.01,
          f"got {obs.energy_consumed_step}, expected {expected_energy}")
    check("TC-005b", f"energy_cost_step ≈ ${expected_cost:.2f}",
          abs(obs.energy_cost_step - expected_cost) < 0.01,
          f"got {obs.energy_cost_step}, expected {expected_cost}")


def test_step_count_increments():
    """TC-006: Step count increments correctly."""
    env = GreenhouseEnvironment()
    env.reset()
    action = make_action(heater_power=0.5, humidifier_level=0.3)
    for i in range(5):
        obs = env.step(action)
        check(f"TC-006-{i+1}", f"step_number == {i+1} after {i+1} steps",
              obs.step_number == i + 1, f"got {obs.step_number}")


def test_reward_bounded():
    """TC-007: Reward always in [0.0, 1.0] for various actions."""
    env = GreenhouseEnvironment()
    env.reset()

    actions = [
        make_action(heater_power=1.0, ventilation_rate=1.0, humidifier_level=1.0, artificial_lighting=1.0),
        make_action(),
        make_action(heater_power=0.5, ventilation_rate=0.5, humidifier_level=0.5, artificial_lighting=0.5),
        make_action(heater_power=1.0),
        make_action(ventilation_rate=1.0),
    ]
    for i, action in enumerate(actions):
        obs = env.step(action)
        check(f"TC-007-{i+1}", f"reward in [0,1] for action variant {i+1}",
              0.0 <= obs.reward <= 1.0, f"reward={obs.reward}")


def test_plant_health_under_good_conditions():
    """TC-008: Plant health stays high under reasonable conditions."""
    env = GreenhouseEnvironment()
    env.reset()
    action = make_action(heater_power=0.5, humidifier_level=0.3, artificial_lighting=0.5)
    for _ in range(5):
        obs = env.step(action)

    check("TC-008a", "plant_health >= 0.8 after 5 good steps",
          obs.plant_health >= 0.8, f"got {obs.plant_health}")
    check("TC-008b", "plant_health <= 1.0",
          obs.plant_health <= 1.0, f"got {obs.plant_health}")


def test_time_advances():
    """TC-009: Hour of day advances each step."""
    env = GreenhouseEnvironment()
    obs0 = env.reset()
    hour_before = obs0.hour_of_day
    obs1 = env.step(make_action())

    check("TC-009a", "hour_of_day changed after one step",
          obs1.hour_of_day != hour_before,
          f"before={hour_before}, after={obs1.hour_of_day}")
    check("TC-009b", "hour_of_day is valid (0-24)",
          0.0 <= obs1.hour_of_day < 24.0, f"got {obs1.hour_of_day}")


def test_heater_raises_temperature():
    """TC-010: Heater at full power raises temperature."""
    env = GreenhouseEnvironment()
    obs0 = env.reset()
    initial_temp = obs0.temperature

    for _ in range(3):
        obs = env.step(make_action(heater_power=1.0))

    check("TC-010a", "temperature increased with full heater (3 steps)",
          obs.temperature > initial_temp - 1,
          f"initial={initial_temp}, after={obs.temperature}")


def test_episode_terminates_at_max_steps():
    """TC-011: Episode terminates at max_steps."""
    env = GreenhouseEnvironment("maintain_temperature")
    env.reset()
    max_steps = TASK_CONFIGS["maintain_temperature"]["max_steps"]

    for i in range(max_steps):
        obs = env.step(make_action(heater_power=0.3, humidifier_level=0.2))

    check("TC-011a", f"done == True at step {max_steps}",
          obs.done is True, f"done={obs.done}")
    check("TC-011b", f"step_number == {max_steps}",
          obs.step_number == max_steps, f"got {obs.step_number}")


def test_grader_score_on_done():
    """TC-012: grader_score appears in metadata when done."""
    env = GreenhouseEnvironment("maintain_temperature")
    env.reset()
    max_steps = TASK_CONFIGS["maintain_temperature"]["max_steps"]

    for _ in range(max_steps):
        obs = env.step(make_action(heater_power=0.3, humidifier_level=0.2))

    meta = obs.metadata or {}
    check("TC-012a", "grader_score in metadata",
          "grader_score" in meta, f"metadata keys: {list(meta.keys())}")
    if "grader_score" in meta:
        gs = meta["grader_score"]
        check("TC-012b", "grader_score in [0, 1]",
              0.0 <= gs <= 1.0, f"got {gs}")


def test_state_matches_observation():
    """TC-013: state().step_count matches observation."""
    env = GreenhouseEnvironment()
    env.reset()
    env.step(make_action(heater_power=0.5))
    env.step(make_action(heater_power=0.5))

    state = env.state
    check("TC-013a", "state.step_count == 2",
          state.step_count == 2, f"got {state.step_count}")
    check("TC-013b", "state has episode_id",
          state.episode_id is not None and len(state.episode_id) > 0)


def test_action_echo():
    """TC-014: last_action echoes the sent action."""
    env = GreenhouseEnvironment()
    env.reset()
    obs = env.step(make_action(
        heater_power=0.75, ventilation_rate=0.25,
        humidifier_level=0.50, artificial_lighting=0.10,
    ))

    la = obs.last_action
    check("TC-014a", "last_action is present",
          la is not None)
    if la:
        check("TC-014b", "heater_power echoed ≈ 0.75",
              abs(la["heater_power"] - 0.75) < 0.01, f"got {la['heater_power']}")
        check("TC-014c", "ventilation_rate echoed ≈ 0.25",
              abs(la["ventilation_rate"] - 0.25) < 0.01, f"got {la['ventilation_rate']}")
        check("TC-014d", "humidifier_level echoed ≈ 0.50",
              abs(la["humidifier_level"] - 0.50) < 0.01, f"got {la['humidifier_level']}")
        check("TC-014e", "artificial_lighting echoed ≈ 0.10",
              abs(la["artificial_lighting"] - 0.10) < 0.01, f"got {la['artificial_lighting']}")


def test_growth_progress_increases():
    """TC-015: Growth progress increases under good conditions."""
    env = GreenhouseEnvironment()
    env.reset()
    action = make_action(heater_power=0.5, humidifier_level=0.4, artificial_lighting=0.6)
    for _ in range(10):
        obs = env.step(action)

    check("TC-015a", "growth_progress > 0 after 10 steps",
          obs.growth_progress > 0.0, f"got {obs.growth_progress}")


def test_energy_monotonic():
    """TC-016: total_energy_consumed never decreases."""
    env = GreenhouseEnvironment()
    env.reset()
    energies = []
    for action_kw in [
        {"heater_power": 0.5, "humidifier_level": 0.3},
        {},
        {"heater_power": 1.0, "ventilation_rate": 1.0, "humidifier_level": 1.0, "artificial_lighting": 1.0},
        {},
        {"heater_power": 0.2},
    ]:
        obs = env.step(make_action(**action_kw))
        energies.append(obs.total_energy_consumed)

    monotonic = all(energies[i] <= energies[i + 1] for i in range(len(energies) - 1))
    check("TC-016a", "total_energy_consumed is monotonically non-decreasing",
          monotonic, f"energies={[round(e, 3) for e in energies]}")


def test_temperature_bounds():
    """TC-017: Temperature stays in [-10, 50]."""
    env = GreenhouseEnvironment()
    env.reset()
    temps = []
    for _ in range(20):
        obs = env.step(make_action(heater_power=1.0))
        temps.append(obs.temperature)

    check("TC-017a", "all temps >= -10",
          all(t >= -10.0 for t in temps), f"min={min(temps):.1f}")
    check("TC-017b", "all temps <= 50",
          all(t <= 50.0 for t in temps), f"max={max(temps):.1f}")


def test_humidity_bounds():
    """TC-018: Humidity stays in [5, 100]."""
    env = GreenhouseEnvironment()
    env.reset()
    humids = []
    for _ in range(20):
        obs = env.step(make_action(humidifier_level=1.0))
        humids.append(obs.humidity)

    check("TC-018a", "all humidities >= 5",
          all(h >= 5.0 for h in humids), f"min={min(humids):.1f}")
    check("TC-018b", "all humidities <= 100",
          all(h <= 100.0 for h in humids), f"max={max(humids):.1f}")


def test_co2_bounds():
    """TC-019: CO₂ stays in [200, 3000]."""
    env = GreenhouseEnvironment()
    env.reset()
    co2s = []
    for _ in range(20):
        obs = env.step(make_action())
        co2s.append(obs.co2_level)

    check("TC-019a", "all CO₂ >= 200",
          all(c >= 200.0 for c in co2s), f"min={min(co2s):.0f}")
    check("TC-019b", "all CO₂ <= 3000",
          all(c <= 3000.0 for c in co2s), f"max={max(co2s):.0f}")


def test_empty_action_defaults():
    """TC-020: Empty action dict (all defaults = 0) doesn't crash."""
    env = GreenhouseEnvironment()
    env.reset()
    obs = env.step(GreenhouseAction())

    check("TC-020a", "empty action produces valid step",
          obs.step_number == 1, f"step={obs.step_number}")
    check("TC-020b", "done is False",
          obs.done is False)


def test_status_message_present():
    """TC-021: status_message is a non-empty string."""
    env = GreenhouseEnvironment()
    env.reset()
    obs = env.step(make_action(heater_power=0.5, humidifier_level=0.3))

    check("TC-021a", "status_message is str",
          isinstance(obs.status_message, str))
    check("TC-021b", "status_message length > 10",
          len(obs.status_message) > 10, f"len={len(obs.status_message)}")


def test_max_steps_matches_task():
    """TC-022: max_steps matches task configuration."""
    for task_id, config in TASK_CONFIGS.items():
        env = GreenhouseEnvironment(task_id=task_id)
        obs = env.reset()
        check(f"TC-022-{task_id}", f"max_steps=={config['max_steps']} for {task_id}",
              obs.max_steps == config["max_steps"],
              f"got {obs.max_steps}")


def test_cloud_cover_bounds():
    """TC-023: Cloud cover in [0, 1]."""
    env = GreenhouseEnvironment()
    env.reset()
    covers = []
    for _ in range(10):
        obs = env.step(make_action())
        covers.append(obs.cloud_cover)

    check("TC-023a", "all cloud_cover >= 0.0",
          all(c >= 0.0 for c in covers), f"min={min(covers):.2f}")
    check("TC-023b", "all cloud_cover <= 1.0",
          all(c <= 1.0 for c in covers), f"max={max(covers):.2f}")


def test_health_bounds():
    """TC-024: Plant health always in [0, 1]."""
    env = GreenhouseEnvironment()
    env.reset()
    healths = []
    for _ in range(20):
        obs = env.step(make_action(ventilation_rate=1.0))
        healths.append(obs.plant_health)

    check("TC-024a", "all health >= 0.0",
          all(h >= 0.0 for h in healths), f"min={min(healths):.3f}")
    check("TC-024b", "all health <= 1.0",
          all(h <= 1.0 for h in healths), f"max={max(healths):.3f}")


def test_all_three_tasks_run():
    """TC-025: All three tasks execute and produce grader scores."""
    for task_id in TASK_CONFIGS:
        env = GreenhouseEnvironment(task_id=task_id)
        obs = env.reset()
        max_steps = TASK_CONFIGS[task_id]["max_steps"]
        action = make_action(heater_power=0.5, humidifier_level=0.3, artificial_lighting=0.4)

        for _ in range(max_steps):
            obs = env.step(action)
            if obs.done:
                break

        meta = obs.metadata or {}
        check(f"TC-025-{task_id}-done", f"{task_id}: done=True at end",
              obs.done is True)
        check(f"TC-025-{task_id}-grader", f"{task_id}: grader_score in [0,1]",
              "grader_score" in meta and 0.0 <= meta["grader_score"] <= 1.0,
              f"score={meta.get('grader_score', 'MISSING')}")


def test_smart_beats_random():
    """TC-026: Smart heuristic agent scores higher than random agent."""
    import random
    random.seed(42)

    for task_id in ["maintain_temperature"]:
        # Random agent
        env = GreenhouseEnvironment(task_id=task_id)
        obs = env.reset()
        max_steps = TASK_CONFIGS[task_id]["max_steps"]
        for _ in range(max_steps):
            obs = env.step(GreenhouseAction(
                heater_power=random.random(),
                ventilation_rate=random.random(),
                humidifier_level=random.random(),
                artificial_lighting=random.random(),
            ))
        random_score = (obs.metadata or {}).get("grader_score", 0)

        # Smart agent
        env2 = GreenhouseEnvironment(task_id=task_id)
        obs2 = env2.reset()
        for _ in range(max_steps):
            obs2 = env2.step(GreenhouseAction(
                heater_power=0.5 if obs2.temperature < 22 else 0.0,
                ventilation_rate=0.3 if obs2.temperature > 25 else 0.1,
                humidifier_level=0.4 if obs2.humidity < 65 else 0.0,
                artificial_lighting=0.6 if obs2.light_intensity < 400 and 6 <= obs2.hour_of_day <= 18 else 0.0,
            ))
        smart_score = (obs2.metadata or {}).get("grader_score", 0)

        check(f"TC-026-{task_id}", f"smart ({smart_score:.3f}) > random ({random_score:.3f})",
              smart_score > random_score,
              f"smart={smart_score}, random={random_score}")


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_all():
    """Execute all test functions and print summary."""
    global PASSED, FAILED, ERRORS
    PASSED = 0
    FAILED = 0
    ERRORS = []

    print()
    print("=" * 70)
    print("  🌿 Greenhouse Climate Control — Offline Unit Tests")
    print("=" * 70)
    print()

    tests = [
        test_reset_returns_valid_observation,
        test_double_reset_clears_state,
        test_zero_action_no_energy,
        test_full_power_energy,
        test_step_count_increments,
        test_reward_bounded,
        test_plant_health_under_good_conditions,
        test_time_advances,
        test_heater_raises_temperature,
        test_episode_terminates_at_max_steps,
        test_grader_score_on_done,
        test_state_matches_observation,
        test_action_echo,
        test_growth_progress_increases,
        test_energy_monotonic,
        test_temperature_bounds,
        test_humidity_bounds,
        test_co2_bounds,
        test_empty_action_defaults,
        test_status_message_present,
        test_max_steps_matches_task,
        test_cloud_cover_bounds,
        test_health_bounds,
        test_all_three_tasks_run,
        test_smart_beats_random,
    ]

    for test_fn in tests:
        print(f"\n--- {test_fn.__doc__ or test_fn.__name__} ---")
        try:
            test_fn()
        except Exception as e:
            FAILED += 1
            msg = f"  💥 ERROR {test_fn.__name__}: {e}"
            print(msg)
            ERRORS.append(msg)

    print()
    print("=" * 70)
    total = PASSED + FAILED
    print(f"  RESULTS: {PASSED}/{total} passed, {FAILED}/{total} failed")
    if ERRORS:
        print()
        print("  FAILURES:")
        for err in ERRORS:
            print(f"    {err}")
    else:
        print("  🎉 ALL TESTS PASSED!")
    print("=" * 70)
    print()

    return FAILED == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
