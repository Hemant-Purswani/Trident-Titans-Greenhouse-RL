"""
Greenhouse Climate Control — Integration API Test Runner.

Tests the running FastAPI server at http://127.0.0.1:8000 using the
native GreenhouseEnv client (which uses persistent WebSockets under the hood).
Also validates the stateless /health HTTP endpoint using requests.

Usage:
    1. Start server:  uvicorn server.app:app --host 0.0.0.0 --port 8000
    2. Run tests:     python tests/test_api.py
"""

import sys
import time
import requests
from pathlib import Path
from typing import List

# Ensure we can import from the greenhouse package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import GreenhouseEnv
from models import GreenhouseAction

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_URL = "http://127.0.0.1:8000"
MAX_RETRIES = 5
RETRY_DELAY = 1.0  # seconds


# ─── Server Connection ──────────────────────────────────────────────────────

def wait_for_server(url: str = BASE_URL, retries: int = MAX_RETRIES) -> bool:
    """Wait for the server to come online. Returns True if ready."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                print(f"  ✅ Server is ready at {url}")
                return True
        except requests.ConnectionError:
            pass
        print(f"  ⏳ Waiting for server... ({attempt}/{retries})")
        time.sleep(RETRY_DELAY)
    return False


# ─── Test Tracking ───────────────────────────────────────────────────────────

PASSED = 0
FAILED = 0
ERRORS: List[str] = []

def check(test_id: str, desc: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ✅ PASS  {test_id}: {desc}")
    else:
        FAILED += 1
        msg = f"  ❌ FAIL  {test_id}: {desc}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        ERRORS.append(msg)


# ─── API Tests ───────────────────────────────────────────────────────────────

def test_api_health():
    """API-01: Health endpoint returns 200."""
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    data = r.json()
    check("API-01a", "/health returns status",
          "status" in data, f"keys={list(data.keys())}")


def test_api_reset_schema():
    """API-02: reset() returns valid observation via client."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        result = env.reset()
        obs = result.observation

        check("API-02-temperature", "observation has temperature", hasattr(obs, 'temperature'))
        check("API-02-humidity", "observation has humidity", hasattr(obs, 'humidity'))
        check("API-02-co2_level", "observation has co2_level", hasattr(obs, 'co2_level'))
        check("API-02-task_id", "observation has task_id", hasattr(obs, 'task_id'))
        check("API-02-step_number", "observation has step_number", hasattr(obs, 'step_number'))


def test_api_reset_values():
    """API-03: reset() returns correct initial values."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        result = env.reset()
        obs = result.observation

        check("API-03a", "step_number == 0", obs.step_number == 0, f"got {obs.step_number}")
        check("API-03b", "plant_health == 1.0", obs.plant_health == 1.0, f"got {obs.plant_health}")
        check("API-03c", "growth_progress == 0.0", obs.growth_progress == 0.0, f"got {obs.growth_progress}")
        check("API-03d", "done == False", result.done is False, f"got {result.done}")


def test_api_step_advances():
    """API-04: step() advances step_number and returns reward."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        env.reset()
        action = GreenhouseAction(heater_power=0.5, ventilation_rate=0.1)
        result = env.step(action)
        obs = result.observation

        check("API-04a", "step_number == 1 after 1 step", obs.step_number == 1, f"got {obs.step_number}")
        
        reward = result.reward
        check("API-04b", "reward is a float", isinstance(reward, float), f"got type {type(reward)}")
        check("API-04c", "reward in [0, 1]", 0.0 <= reward <= 1.0, f"got {reward}")


def test_api_step_energy():
    """API-05: step() energy consumption matches actions."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        env.reset()
        action = GreenhouseAction(heater_power=1.0, ventilation_rate=1.0, humidifier_level=1.0, artificial_lighting=1.0)
        result = env.step(action)
        obs = result.observation

        energy = obs.energy_consumed_step
        check("API-05a", "energy_consumed_step > 0 with full power", energy > 0, f"got {energy}")
        check("API-05b", "energy_consumed_step ≈ 4.0", abs(energy - 4.0) < 0.1, f"got {energy}")


def test_api_step_empty_action():
    """API-06: step() with empty action (defaults to zeros)."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        env.reset()
        result = env.step(GreenhouseAction())
        obs = result.observation

        check("API-06a", "step_number == 1 with empty action", obs.step_number == 1)
        check("API-06b", "energy == 0 with empty action", obs.energy_consumed_step == 0.0)


def test_api_state_consistency():
    """API-07: Env client state representation is cohesive."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        env.reset()
        env.step(GreenhouseAction(heater_power=0.5))
        env.step(GreenhouseAction(heater_power=0.5))

        state = env.state()
        check("API-07a", "state.step_count == 2", state.step_count == 2, f"got {state.step_count}")
        check("API-07c", "state has episode_id", state.episode_id is not None)


def test_api_multi_step_episode():
    """API-08: Run 24 steps (full episode), done should be True."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        env.reset()
        rewards = []
        action = GreenhouseAction(heater_power=0.3, ventilation_rate=0.1, humidifier_level=0.2)
        
        for _ in range(24):
            result = env.step(action)
            rewards.append(result.reward)

        obs = result.observation
        check("API-08a", "done == True after 24 steps", result.done is True, f"got {result.done}")
        check("API-08b", "all 24 rewards in [0, 1]", all(0.0 <= r <= 1.0 for r in rewards))
        check("API-08c", "step_number == 24", obs.step_number == 24, f"got {obs.step_number}")


def test_api_action_echo():
    """API-09: last_action in observation echoes sent action."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        env.reset()
        action = GreenhouseAction(heater_power=0.8, ventilation_rate=0.2, humidifier_level=0.6, artificial_lighting=0.4)
        result = env.step(action)

        la = result.observation.last_action
        check("API-09a", "last_action is present", la is not None)
        if la:
             check("API-09-heater_power", "last_action.heater_power ≈ 0.8", abs(la["heater_power"] - 0.8) < 0.01)
             check("API-09-ventilation_rate", "last_action.ventilation_rate ≈ 0.2", abs(la["ventilation_rate"] - 0.2) < 0.01)


def test_api_reset_after_episode():
    """API-10: Reset after completed episode produces clean state."""
    with GreenhouseEnv(base_url=BASE_URL).sync() as env:
        env.reset()
        for _ in range(24):
             env.step(GreenhouseAction(heater_power=0.3))

        result = env.reset()
        obs = result.observation

        check("API-10a", "step_number == 0 after reset", obs.step_number == 0)
        check("API-10b", "plant_health == 1.0 after reset", obs.plant_health == 1.0)
        check("API-10c", "done == False after reset", result.done is False)


# ─── Runner ──────────────────────────────────────────────────────────────────

def run_all() -> bool:
    global PASSED, FAILED, ERRORS
    PASSED = 0
    FAILED = 0
    ERRORS = []

    print()
    print("=" * 70)
    print("  🌿 Greenhouse Climate Control — Integration API Tests (EnvClient)")
    print(f"  Server: {BASE_URL}")
    print("=" * 70)
    print()

    if not wait_for_server():
        print()
        print("  ❌ Server is not running. Aborting.")
        return False

    tests = [
        test_api_health,
        test_api_reset_schema,
        test_api_reset_values,
        test_api_step_advances,
        test_api_step_energy,
        test_api_step_empty_action,
        test_api_state_consistency,
        test_api_multi_step_episode,
        test_api_action_echo,
        test_api_reset_after_episode,
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
        print("  FAILURES:")
        for err in ERRORS:
            print(f"    {err}")
    else:
        print("  🎉 ALL API TESTS PASSED!")
    print("=" * 70)
    print()

    return FAILED == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
