# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Greenhouse Climate Control Environment Implementation.

A realistic greenhouse simulation where an agent controls heater, ventilation,
humidifier, and artificial lighting to optimize crop growth while minimizing
energy consumption. Features realistic physics, stochastic weather, day/night
cycles, and sensor noise.

Tasks:
    1. maintain_temperature (easy)   — Keep temp in range for 1 day
    2. optimize_growth     (medium)  — Maximize growth over 3 days
    3. weather_resilience  (hard)    — Survive 7 days of extreme weather
"""

import math
import random
from typing import Dict, Any, Optional, List, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import GreenhouseAction, GreenhouseObservation, GreenhouseState
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models import GreenhouseAction, GreenhouseObservation, GreenhouseState


# ─── Optimal crop ranges ────────────────────────────────────────────────────
OPTIMAL_TEMP_MIN, OPTIMAL_TEMP_MAX = 20.0, 26.0
SURVIVABLE_TEMP_MIN, SURVIVABLE_TEMP_MAX = 10.0, 38.0

OPTIMAL_HUMIDITY_MIN, OPTIMAL_HUMIDITY_MAX = 60.0, 80.0
SURVIVABLE_HUMIDITY_MIN, SURVIVABLE_HUMIDITY_MAX = 30.0, 95.0

OPTIMAL_CO2_MIN, OPTIMAL_CO2_MAX = 800.0, 1200.0
SURVIVABLE_CO2_MIN, SURVIVABLE_CO2_MAX = 300.0, 2000.0

OPTIMAL_LIGHT_MIN, OPTIMAL_LIGHT_MAX = 400.0, 800.0  # µmol/m²/s PAR

# ─── Physics constants ──────────────────────────────────────────────────────
HEATER_MAX_DELTA = 5.0       # Max temp increase per step at full power (°C)
VENTILATION_EXCHANGE = 0.4   # Rate of air exchange with outside
HUMIDIFIER_MAX_DELTA = 8.0   # Max humidity increase per step (%)
THERMAL_INERTIA = 0.85       # How much indoor temp resists change
CO2_NATURAL_DECAY = 5.0      # CO₂ ppm lost per step to leakage
CO2_PLANT_UPTAKE = 3.0       # CO₂ ppm consumed by plants per step at optimal conditions
MAX_ARTIFICIAL_LIGHT = 600.0 # µmol/m²/s from grow lights at full power
MAX_NATURAL_LIGHT = 900.0    # Peak natural light µmol/m²/s at solar noon

# ─── Energy costs ────────────────────────────────────────────────────────────
HEATER_ENERGY_KWH = 2.0      # kWh at full power per step
VENTILATION_ENERGY_KWH = 0.3
HUMIDIFIER_ENERGY_KWH = 0.2
LIGHTING_ENERGY_KWH = 1.5
ENERGY_PRICE_PER_KWH = 0.15  # $/kWh

# ─── Task configurations ────────────────────────────────────────────────────
TASK_CONFIGS = {
    "maintain_temperature": {
        "max_steps": 24,
        "description": "Keep greenhouse temperature in 20-26°C optimal range for 24 hours.",
        "difficulty": "easy",
        "extreme_weather": False,
        "weather_volatility": 0.3,
    },
    "optimize_growth": {
        "max_steps": 72,
        "description": "Maximize crop growth over 3 days while minimizing energy consumption.",
        "difficulty": "medium",
        "extreme_weather": False,
        "weather_volatility": 0.5,
    },
    "weather_resilience": {
        "max_steps": 168,
        "description": "Maintain all optimal conditions through 7 days of extreme weather events.",
        "difficulty": "hard",
        "extreme_weather": True,
        "weather_volatility": 0.9,
    },
    "resource_efficiency_master": {
        "max_steps": 240,
        "description": "Maximize growth over 10 days while adhering to a strict Net Zero energy budget.",
        "difficulty": "expert",
        "extreme_weather": True,
        "weather_volatility": 1.2,
    },
}

# ─── Reward weights per task ────────────────────────────────────────────────
REWARD_WEIGHTS = {
    "maintain_temperature": {
        "temperature": 0.7,
        "humidity": 0.1,
        "light": 0.05,
        "co2": 0.05,
        "energy": 0.05,
        "stability": 0.05,
    },
    "optimize_growth": {
        "temperature": 0.25,
        "humidity": 0.2,
        "light": 0.15,
        "co2": 0.1,
        "energy": 0.15,
        "stability": 0.15,
    },
    "weather_resilience": {
        "temperature": 0.2,
        "humidity": 0.2,
        "light": 0.15,
        "co2": 0.1,
        "energy": 0.1,
        "stability": 0.25,
    },
    "resource_efficiency_master": {
        "temperature": 0.3,
        "humidity": 0.1,
        "light": 0.1,
        "co2": 0.1,
        "energy": 0.3,
        "stability": 0.1,
    },
}


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def _range_score(value: float, opt_min: float, opt_max: float,
                 surv_min: float, surv_max: float) -> float:
    """
    Score how well a value falls within optimal/survivable ranges.

    Returns:
        1.0 if within optimal range
        0.0-1.0 if between optimal and survivable boundary (linear decay)
        0.0 if outside survivable range
    """
    if opt_min <= value <= opt_max:
        return 1.0
    elif value < opt_min:
        if value <= surv_min:
            return 0.0
        return (value - surv_min) / (opt_min - surv_min)
    else:  # value > opt_max
        if value >= surv_max:
            return 0.0
        return (surv_max - value) / (surv_max - opt_max)


class GreenhouseEnvironment(Environment):
    """
    Greenhouse Climate Control Environment.

    Simulates a realistic greenhouse with:
    - Thermal dynamics (heater, ventilation, outdoor coupling)
    - Humidity control (humidifier, ventilation exchange)
    - CO₂ management (natural decay, plant uptake)
    - Light management (day/night cycle, weather, artificial lights)
    - Stochastic weather (temperature, cloud cover, humidity)
    - Sensor noise
    - 3 difficulty-graded tasks with programmatic graders

    The agent controls 4 actuators each timestep (1 step = 1 hour):
        heater_power, ventilation_rate, humidifier_level, artificial_lighting
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    TASKS: List[str] = [
        "maintain_temperature",
        "optimize_growth",
        "weather_resilience",
        "resource_efficiency_master",
    ]

    def __init__(self, task_id: str = "maintain_temperature"):
        """Initialize the greenhouse environment."""
        self._task_id = task_id if task_id in TASK_CONFIGS else "maintain_temperature"
        self._config = TASK_CONFIGS[self._task_id]
        self._state = GreenhouseState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task_id,
        )

        # Internal simulation state
        self._temperature: float = 22.0
        self._humidity: float = 65.0
        self._co2: float = 800.0
        self._light: float = 0.0

        # Weather state
        self._outside_temp: float = 15.0
        self._outside_humidity: float = 50.0
        self._cloud_cover: float = 0.3
        self._weather_trend: float = 0.0  # random walk component

        # Time
        self._hour: float = 8.0  # Start at 8 AM
        self._day: int = 1

        # Crop state
        self._plant_health: float = 1.0
        self._growth_progress: float = 0.0

        # Energy tracking
        self._total_energy: float = 0.0
        self._total_cost: float = 0.0

        # Reward tracking
        self._total_reward: float = 0.0
        self._prev_temperature: float = 22.0
        self._prev_humidity: float = 65.0

        # Per-step tracking for graders
        self._step_scores: List[float] = []
        self._temp_in_range_count: int = 0

        # Random seed for reproducibility within an episode
        self._rng = random.Random()

    def reset(self, task_id: Optional[str] = None) -> GreenhouseObservation:
        """
        Reset the environment for a new episode.

        Args:
            task_id: Optional task to activate. Defaults to current task.

        Returns:
            Initial GreenhouseObservation
        """
        if task_id and task_id in TASK_CONFIGS:
            self._task_id = task_id
            self._config = TASK_CONFIGS[self._task_id]

        self._state = GreenhouseState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task_id,
        )

        # Initial conditions depend on task difficulty
        if self._task_id == "maintain_temperature":
            # Easy: start close to range but slightly cool
            self._temperature = 17.0 + self._rng.uniform(0, 4)
            self._humidity = 58.0 + self._rng.uniform(-5, 15)
            self._co2 = 750.0 + self._rng.uniform(-100, 200)
            self._outside_temp = 10.0 + self._rng.uniform(-3, 8)
            self._cloud_cover = self._rng.uniform(0.1, 0.5)
        elif self._task_id == "optimize_growth":
            # Medium: start at night, cold, low humidity — agent must fix everything
            self._temperature = 14.0 + self._rng.uniform(-3, 3)
            self._humidity = 45.0 + self._rng.uniform(-5, 10)
            self._co2 = 600.0 + self._rng.uniform(-100, 150)
            self._outside_temp = 8.0 + self._rng.uniform(-4, 5)
            self._cloud_cover = self._rng.uniform(0.4, 0.9)
        else:  # weather_resilience (hard)
            # Hard: start with a cold snap already in progress
            self._temperature = 12.0 + self._rng.uniform(-4, 4)
            self._humidity = 40.0 + self._rng.uniform(-10, 20)
            self._co2 = 550.0 + self._rng.uniform(-100, 200)
            self._outside_temp = self._rng.choice([4.0, 32.0])
            self._cloud_cover = self._rng.uniform(0.0, 1.0)

        self._outside_humidity = 40.0 + self._rng.uniform(0, 30)
        self._weather_trend = 0.0

        # Time: start at 8 AM
        self._hour = 8.0
        self._day = 1

        # Compute initial light
        self._light = self._compute_natural_light()

        # Crop state
        self._plant_health = 1.0
        self._growth_progress = 0.0

        # Energy
        self._total_energy = 0.0
        self._total_cost = 0.0

        # Reward tracking
        self._total_reward = 0.0
        self._prev_temperature = self._temperature
        self._prev_humidity = self._humidity

        # Grading
        self._step_scores = []
        self._temp_in_range_count = 0

        return self._build_observation(
            energy_step=0.0,
            cost_step=0.0,
            reward=0.0,
            done=False,
            last_action=None,
        )

    def step(self, action: GreenhouseAction) -> GreenhouseObservation:  # type: ignore[override]
        """
        Execute one timestep (1 hour) of greenhouse simulation.

        Args:
            action: GreenhouseAction with 4 control values

        Returns:
            GreenhouseObservation with updated state
        """
        self._state.step_count += 1

        # Clamp action values to valid range
        heater = _clamp(action.heater_power, 0.0, 1.0)
        vent = _clamp(action.ventilation_rate, 0.0, 1.0)
        humid = _clamp(action.humidifier_level, 0.0, 1.0)
        light_ctrl = _clamp(action.artificial_lighting, 0.0, 1.0)

        # Store previous values for stability calculation
        self._prev_temperature = self._temperature
        self._prev_humidity = self._humidity

        # ── Step 1: Update weather ──────────────────────────────────
        self._update_weather()

        # ── Step 2: Physics simulation ──────────────────────────────
        self._simulate_temperature(heater, vent)
        self._simulate_humidity(humid, vent)
        self._simulate_co2()
        self._simulate_light(light_ctrl)

        # ── Step 3: Add sensor noise ────────────────────────────────
        noise_temp = self._rng.gauss(0, 0.3)
        noise_humid = self._rng.gauss(0, 0.5)
        noise_co2 = self._rng.gauss(0, 5.0)
        noise_light = self._rng.gauss(0, 10.0)

        # ── Step 4: Compute energy consumption ──────────────────────
        energy_step = (
            heater * HEATER_ENERGY_KWH
            + vent * VENTILATION_ENERGY_KWH
            + humid * HUMIDIFIER_ENERGY_KWH
            + light_ctrl * LIGHTING_ENERGY_KWH
        )
        cost_step = energy_step * ENERGY_PRICE_PER_KWH
        self._total_energy += energy_step
        self._total_cost += cost_step

        # ── Step 5: Update crop state ───────────────────────────────
        self._update_crop()

        # ── Step 6: Advance time ────────────────────────────────────
        self._hour += 1.0
        if self._hour >= 24.0:
            self._hour -= 24.0
            self._day += 1

        # ── Step 7: Compute reward ──────────────────────────────────
        raw_reward = self._compute_reward(energy_step)
        # Hyper-fix: Map reward strictly to (0.01, 0.99)
        reward = 0.01 + 0.98 * _clamp(raw_reward, 0.0, 1.0)
        self._total_reward += reward

        # Track for grading
        self._step_scores.append(reward)
        if OPTIMAL_TEMP_MIN <= self._temperature <= OPTIMAL_TEMP_MAX:
            self._temp_in_range_count += 1

        # ── Step 8: Check termination ───────────────────────────────
        done = (
            self._state.step_count >= self._config["max_steps"]
            or self._plant_health <= 0.0
        )

        # Update extended state
        self._state.total_reward = self._total_reward
        self._state.total_energy = self._total_energy
        self._state.plant_health = self._plant_health
        self._state.growth_progress = self._growth_progress

        last_action = {
            "heater_power": round(heater, 3),
            "ventilation_rate": round(vent, 3),
            "humidifier_level": round(humid, 3),
            "artificial_lighting": round(light_ctrl, 3),
        }

        obs = self._build_observation(
            energy_step=energy_step,
            cost_step=cost_step,
            reward=reward,
            done=done,
            last_action=last_action,
            noise_temp=noise_temp,
            noise_humid=noise_humid,
            noise_co2=noise_co2,
            noise_light=noise_light,
        )

        # If done, compute final grader score and attach to metadata
        if done:
            grader_score = self.grader()
            obs.metadata = obs.metadata or {}
            obs.metadata["grader_score"] = round(grader_score, 4)
            obs.metadata["task_id"] = self._task_id
            obs.metadata["final"] = True

        return obs

    @property
    def state(self) -> GreenhouseState:
        """Get the current environment state."""
        return self._state

    # ─── Physics Simulation ──────────────────────────────────────────────────

    def _simulate_temperature(self, heater: float, ventilation: float) -> None:
        """
        Update greenhouse temperature based on heater, ventilation, and outdoor coupling.

        Physics model:
            - Heater adds heat proportional to power
            - Ventilation mixes indoor/outdoor air
            - Natural thermal drift toward outside temp (conduction)
            - Time-of-day effects (radiative heating during day)
        """
        # Heater effect
        heater_delta = heater * HEATER_MAX_DELTA

        # Ventilation exchange with outside
        vent_delta = ventilation * VENTILATION_EXCHANGE * (self._outside_temp - self._temperature)

        # Natural conduction loss/gain toward outside temp
        conduction = 0.05 * (self._outside_temp - self._temperature)

        # Solar radiative heating (during daytime)
        solar_factor = max(0, math.sin(math.pi * (self._hour - 6) / 12)) if 6 <= self._hour <= 18 else 0
        solar_heating = solar_factor * (1 - self._cloud_cover) * 1.5

        # Apply changes with thermal inertia
        delta = heater_delta + vent_delta + conduction + solar_heating
        self._temperature = self._temperature * THERMAL_INERTIA + (self._temperature + delta) * (1 - THERMAL_INERTIA)
        self._temperature += delta * (1 - THERMAL_INERTIA)

        # Simplified: apply delta with inertia damping
        self._temperature = self._prev_temperature + delta * 0.6

        # Add small process noise
        self._temperature += self._rng.gauss(0, 0.15)

        # Physical bounds
        self._temperature = _clamp(self._temperature, -10.0, 50.0)

    def _simulate_humidity(self, humidifier: float, ventilation: float) -> None:
        """
        Update greenhouse humidity.

        Physics model:
            - Humidifier adds moisture
            - Ventilation exchanges with outside humidity
            - Temperature coupling (warmer air holds more moisture, relative drops)
        """
        # Humidifier effect
        humid_delta = humidifier * HUMIDIFIER_MAX_DELTA

        # Ventilation exchange
        vent_delta = ventilation * VENTILATION_EXCHANGE * (self._outside_humidity - self._humidity)

        # Temperature-humidity coupling: rapid temp increase lowers relative humidity
        temp_change = self._temperature - self._prev_temperature
        temp_coupling = -temp_change * 1.5  # Warming reduces RH

        # Natural evapotranspiration from plants (small increase)
        evapotranspiration = 0.3 * self._plant_health

        delta = humid_delta + vent_delta + temp_coupling + evapotranspiration
        self._humidity += delta * 0.5  # Damped

        # Add process noise
        self._humidity += self._rng.gauss(0, 0.3)

        # Physical bounds
        self._humidity = _clamp(self._humidity, 5.0, 100.0)

    def _simulate_co2(self) -> None:
        """
        Update CO₂ levels.

        Physics model:
            - Natural leakage decay
            - Plant photosynthesis uptake (proportional to light and health)
            - Small natural replenishment from soil respiration
        """
        # Leakage
        leak = CO2_NATURAL_DECAY * self._rng.uniform(0.8, 1.2)

        # Plant uptake — higher during photosynthesis (needs light)
        light_factor = min(self._light / OPTIMAL_LIGHT_MAX, 1.0) if self._light > 50 else 0
        uptake = CO2_PLANT_UPTAKE * light_factor * self._plant_health

        # Soil respiration (natural CO₂ release)
        respiration = 4.0 * self._rng.uniform(0.8, 1.2)

        self._co2 += respiration - leak - uptake

        # Add process noise
        self._co2 += self._rng.gauss(0, 3.0)

        # Physical bounds
        self._co2 = _clamp(self._co2, 200.0, 3000.0)

    def _simulate_light(self, artificial: float) -> None:
        """
        Update light intensity.

        Combines natural sunlight (day/night cycle, weather) with artificial lighting.
        """
        natural = self._compute_natural_light()
        artificial_light = artificial * MAX_ARTIFICIAL_LIGHT

        self._light = natural + artificial_light

        # Physical bounds
        self._light = _clamp(self._light, 0.0, 1500.0)

    def _compute_natural_light(self) -> float:
        """
        Compute natural sunlight intensity based on time and weather.

        Uses sinusoidal day/night cycle, modulated by cloud cover.
        Sunrise ~6:00, sunset ~18:00.
        """
        if self._hour < 6 or self._hour > 18:
            return 0.0

        # Sinusoidal: peak at solar noon (12:00)
        solar_angle = math.sin(math.pi * (self._hour - 6) / 12)
        base_light = MAX_NATURAL_LIGHT * max(0, solar_angle)

        # Cloud attenuation
        cloud_factor = 1.0 - (self._cloud_cover * 0.75)  # Clouds block up to 75%

        return base_light * cloud_factor

    # ─── Weather System ──────────────────────────────────────────────────────

    def _update_weather(self) -> None:
        """
        Update outdoor weather conditions stochastically.

        - Temperature: sinusoidal daily cycle + random walk
        - Cloud cover: bounded random walk
        - Humidity: correlated with cloud cover
        """
        volatility = self._config["weather_volatility"]

        # Base outdoor temperature: sinusoidal daily cycle
        # Cooler at night (hour 4), warmer during day (hour 14)
        daily_base = 15.0 + 5.0 * math.sin(math.pi * (self._hour - 4) / 12)

        # Random walk with mean reversion
        self._weather_trend += self._rng.gauss(0, volatility * 1.5)
        self._weather_trend *= 0.95  # Mean reversion

        # Extreme weather events for hard task
        if self._config["extreme_weather"]:
            # Occasional weather shocks
            if self._rng.random() < 0.05:
                shock = self._rng.choice([-12, -8, 8, 12])
                self._weather_trend += shock

        self._outside_temp = daily_base + self._weather_trend
        self._outside_temp = _clamp(self._outside_temp, -5.0, 42.0)

        # Cloud cover: bounded random walk
        cloud_delta = self._rng.gauss(0, 0.08 * volatility)
        self._cloud_cover += cloud_delta
        self._cloud_cover = _clamp(self._cloud_cover, 0.0, 1.0)

        # Extreme weather: occasional rapid cloud changes
        if self._config["extreme_weather"] and self._rng.random() < 0.08:
            self._cloud_cover = self._rng.uniform(0.0, 1.0)

        # Outside humidity correlates with clouds
        base_humidity = 40 + 30 * self._cloud_cover
        self._outside_humidity += self._rng.gauss(0, 2.0 * volatility)
        self._outside_humidity = 0.7 * self._outside_humidity + 0.3 * base_humidity
        self._outside_humidity = _clamp(self._outside_humidity, 20.0, 100.0)

    # ─── Crop Model ──────────────────────────────────────────────────────────

    def _update_crop(self) -> None:
        """
        Update plant health and growth based on current conditions.

        Growth rate depends on how close conditions are to optimal.
        Health degrades when conditions are outside survivable range.
        """
        # Score each condition
        temp_score = _range_score(
            self._temperature,
            OPTIMAL_TEMP_MIN, OPTIMAL_TEMP_MAX,
            SURVIVABLE_TEMP_MIN, SURVIVABLE_TEMP_MAX,
        )
        humid_score = _range_score(
            self._humidity,
            OPTIMAL_HUMIDITY_MIN, OPTIMAL_HUMIDITY_MAX,
            SURVIVABLE_HUMIDITY_MIN, SURVIVABLE_HUMIDITY_MAX,
        )
        co2_score = _range_score(
            self._co2,
            OPTIMAL_CO2_MIN, OPTIMAL_CO2_MAX,
            SURVIVABLE_CO2_MIN, SURVIVABLE_CO2_MAX,
        )

        # Light score (adapted for day/night — during night, no light needed)
        if 6 <= self._hour <= 18:
            light_score = _range_score(
                self._light,
                OPTIMAL_LIGHT_MIN, OPTIMAL_LIGHT_MAX,
                50.0, 1200.0,
            )
        else:
            # At night, light should be low (darkness period is beneficial)
            light_score = 1.0 if self._light < 100 else 0.5

        # Combined condition quality
        condition_quality = (
            0.35 * temp_score
            + 0.25 * humid_score
            + 0.20 * light_score
            + 0.20 * co2_score
        )

        # Growth: proportional to condition quality
        max_growth_per_step = 1.0 / self._config["max_steps"]
        growth_rate = condition_quality * max_growth_per_step * self._plant_health
        self._growth_progress = _clamp(
            self._growth_progress + growth_rate, 0.0, 1.0
        )

        # Health: degrades if conditions are very poor
        if condition_quality < 0.3:
            health_loss = (0.3 - condition_quality) * 0.05
            self._plant_health -= health_loss
        elif condition_quality > 0.6:
            # Slow health recovery
            self._plant_health = min(1.0, self._plant_health + 0.002)

        # Critical damage for extreme conditions
        if self._temperature < SURVIVABLE_TEMP_MIN or self._temperature > SURVIVABLE_TEMP_MAX:
            self._plant_health -= 0.08
        if self._humidity < SURVIVABLE_HUMIDITY_MIN or self._humidity > SURVIVABLE_HUMIDITY_MAX:
            self._plant_health -= 0.03

        self._plant_health = _clamp(self._plant_health, 0.0, 1.0)

    # ─── Reward Function ─────────────────────────────────────────────────────

    def _compute_reward(self, energy_step: float) -> float:
        """
        Compute multi-objective reward for the current step.

        Components:
            - Temperature score: How well temp matches optimal range
            - Humidity score: How well humidity matches optimal range
            - Light score: How well light matches needs
            - CO₂ score: How well CO₂ matches optimal range
            - Energy penalty: Penalize excessive energy use
            - Stability bonus: Reward smooth transitions

        All components are normalized to [0, 1] and weighted per task.
        Final reward is in [0, 1] range.
        """
        weights = REWARD_WEIGHTS[self._task_id]

        # Temperature score
        temp_score = _range_score(
            self._temperature,
            OPTIMAL_TEMP_MIN, OPTIMAL_TEMP_MAX,
            SURVIVABLE_TEMP_MIN, SURVIVABLE_TEMP_MAX,
        )

        # Humidity score
        humid_score = _range_score(
            self._humidity,
            OPTIMAL_HUMIDITY_MIN, OPTIMAL_HUMIDITY_MAX,
            SURVIVABLE_HUMIDITY_MIN, SURVIVABLE_HUMIDITY_MAX,
        )

        # Light score (day/night aware)
        if 6 <= self._hour <= 18:
            light_score = _range_score(
                self._light,
                OPTIMAL_LIGHT_MIN, OPTIMAL_LIGHT_MAX,
                50.0, 1200.0,
            )
        else:
            light_score = 1.0 if self._light < 100 else 0.5

        # CO₂ score
        co2_score = _range_score(
            self._co2,
            OPTIMAL_CO2_MIN, OPTIMAL_CO2_MAX,
            SURVIVABLE_CO2_MIN, SURVIVABLE_CO2_MAX,
        )

        # Energy efficiency (lower is better)
        max_energy = (
            HEATER_ENERGY_KWH + VENTILATION_ENERGY_KWH
            + HUMIDIFIER_ENERGY_KWH + LIGHTING_ENERGY_KWH
        )
        energy_score = 1.0 - min(energy_step / max_energy, 1.0)

        # Stability: penalize large changes
        temp_change = abs(self._temperature - self._prev_temperature)
        humid_change = abs(self._humidity - self._prev_humidity)
        stability_score = max(0, 1.0 - (temp_change / 5.0 + humid_change / 10.0))

        # Weighted sum
        reward = (
            weights["temperature"] * temp_score
            + weights["humidity"] * humid_score
            + weights["light"] * light_score
            + weights["co2"] * co2_score
            + weights["energy"] * energy_score
            + weights["stability"] * stability_score
        )

        # Map range to be strictly between (0, 1) as per hackathon phase 2 requirements
        return 0.01 + 0.98 * _clamp(reward, 0.0, 1.0)

    # ─── Grader ──────────────────────────────────────────────────────────────

    def grader(self, task_id: Optional[str] = None) -> float:
        """
        Compute the final grader score for the episode (0.0 – 1.0).

        Grading criteria vary by task:
            - maintain_temperature: fraction of steps with temp in optimal range
            - optimize_growth: growth_progress * (1 - energy_penalty) * health
            - weather_resilience: weighted(health, growth, energy, stability)

        Mapped to (0.01, 0.99) to satisfy hackathon validation requirements.
        """
        target_task = task_id if task_id in TASK_CONFIGS else self._task_id
        config = TASK_CONFIGS[target_task]
        
        max_steps = config["max_steps"]
        steps_taken = self._state.step_count

        if steps_taken == 0:
            return 0.01

        if target_task == "maintain_temperature":
            # Score = fraction of steps temperature was in optimal range
            score = self._temp_in_range_count / max(steps_taken, 1)
            return 0.01 + 0.98 * _clamp(score, 0.0, 1.0)

        elif target_task == "optimize_growth":
            # Growth quality weighted by energy efficiency and avg step quality
            growth_score = self._growth_progress

            # Average per-step reward reflects ongoing condition quality
            avg_reward = sum(self._step_scores) / max(len(self._step_scores), 1)

            # Energy efficiency: compare to a "reasonable" baseline (1.5 kWh/step)
            reasonable_energy = max_steps * 1.5
            energy_ratio = self._total_energy / max(reasonable_energy, 0.01)
            # Penalty starts when energy exceeds 50% of max possible
            energy_penalty = _clamp((energy_ratio - 0.5) * 0.4, 0.0, 0.4)

            health_factor = self._plant_health

            score = (
                0.40 * growth_score
                + 0.35 * avg_reward
                + 0.25 * health_factor
            ) * (1.0 - energy_penalty)
            return 0.01 + 0.98 * _clamp(score, 0.0, 1.0)

        elif target_task == "weather_resilience":
            # Multi-factor grading
            health_score = self._plant_health
            growth_score = self._growth_progress

            # Average per-step reward as quality metric
            avg_reward = sum(self._step_scores) / max(len(self._step_scores), 1)

            # Survival bonus (finished alive)
            survival = 1.0 if self._plant_health > 0.1 else 0.0

            score = (
                0.30 * health_score
                + 0.25 * growth_score
                + 0.25 * avg_reward
                + 0.20 * survival
            )
            return 0.01 + 0.98 * _clamp(score, 0.0, 1.0)
            
        elif target_task == "resource_efficiency_master":
            # Multi-factor grading with emphasis on energy
            health_score = self._plant_health
            growth_score = self._growth_progress

            # Average per-step reward
            avg_reward = sum(self._step_scores) / max(len(self._step_scores), 1)

            # Strict energy efficiency: compare to target (1.0 kWh/step)
            strict_energy_target = max_steps * 1.0
            energy_ratio = self._total_energy / max(strict_energy_target, 0.01)
            # Steep penalty for exceeding budget
            energy_score = _clamp(1.2 - energy_ratio, 0.0, 1.0)

            score = (
                0.35 * growth_score
                + 0.30 * energy_score
                + 0.20 * health_score
                + 0.15 * avg_reward
            )
            return 0.01 + 0.98 * _clamp(score, 0.0, 1.0)

        return 0.01

    # ─── Observation Building ────────────────────────────────────────────────

    def _build_observation(
        self,
        energy_step: float,
        cost_step: float,
        reward: float,
        done: bool,
        last_action: Optional[Dict[str, float]],
        noise_temp: float = 0.0,
        noise_humid: float = 0.0,
        noise_co2: float = 0.0,
        noise_light: float = 0.0,
    ) -> GreenhouseObservation:
        """Build a GreenhouseObservation from current state + noise."""

        # Status message for LLM
        status = self._build_status_message(energy_step, cost_step, reward)

        return GreenhouseObservation(
            # Indoor (with sensor noise, clamped to physical bounds)
            temperature=round(_clamp(self._temperature + noise_temp, -10.0, 50.0), 1),
            humidity=round(_clamp(self._humidity + noise_humid, 5.0, 100.0), 1),
            co2_level=round(_clamp(self._co2 + noise_co2, 200.0, 3000.0), 0),
            light_intensity=round(max(0, self._light + noise_light), 0),
            # Outdoor
            outside_temperature=round(self._outside_temp, 1),
            outside_humidity=round(self._outside_humidity, 1),
            cloud_cover=round(self._cloud_cover, 2),
            # Time
            hour_of_day=round(self._hour, 1),
            day_number=self._day,
            # Crop
            plant_health=round(self._plant_health, 3),
            growth_progress=round(self._growth_progress, 4),
            # Energy
            energy_consumed_step=round(energy_step, 3),
            total_energy_consumed=round(self._total_energy, 3),
            energy_cost_step=round(cost_step, 4),
            total_energy_cost=round(self._total_cost, 4),
            # Episode
            step_number=self._state.step_count,
            max_steps=self._config["max_steps"],
            task_id=self._task_id,
            # Status
            status_message=status,
            # Action echo
            last_action=last_action,
            # Standard fields
            done=done,
            # Hyper-fix: Ensure observation reward is strictly (0.01, 0.99)
            reward=max(0.01, min(0.99, round(reward, 4))),
            metadata={
                "step": self._state.step_count,
                "task": self._task_id,
                "difficulty": self._config["difficulty"],
            },
        )

    def _build_status_message(self, energy_step: float, cost_step: float,
                              reward: float) -> str:
        """Build a human-readable status message for the LLM agent."""
        time_str = f"{int(self._hour):02d}:00"
        period = "day" if 6 <= self._hour <= 18 else "night"

        # Condition assessment
        conditions = []
        if self._temperature < OPTIMAL_TEMP_MIN:
            conditions.append(f"COLD ({self._temperature:.1f}°C < {OPTIMAL_TEMP_MIN}°C)")
        elif self._temperature > OPTIMAL_TEMP_MAX:
            conditions.append(f"HOT ({self._temperature:.1f}°C > {OPTIMAL_TEMP_MAX}°C)")
        else:
            conditions.append(f"temp OK ({self._temperature:.1f}°C)")

        if self._humidity < OPTIMAL_HUMIDITY_MIN:
            conditions.append(f"DRY ({self._humidity:.1f}%)")
        elif self._humidity > OPTIMAL_HUMIDITY_MAX:
            conditions.append(f"HUMID ({self._humidity:.1f}%)")
        else:
            conditions.append(f"humidity OK ({self._humidity:.1f}%)")

        if self._co2 < OPTIMAL_CO2_MIN:
            conditions.append(f"low CO₂ ({self._co2:.0f}ppm)")
        elif self._co2 > OPTIMAL_CO2_MAX:
            conditions.append(f"high CO₂ ({self._co2:.0f}ppm)")

        if period == "day" and self._light < OPTIMAL_LIGHT_MIN:
            conditions.append(f"low light ({self._light:.0f}µmol)")

        weather = "clear" if self._cloud_cover < 0.3 else "partly cloudy" if self._cloud_cover < 0.7 else "overcast"

        msg = (
            f"[Day {self._day}, {time_str} ({period})] "
            f"Outside: {self._outside_temp:.1f}°C, {weather}. "
            f"Greenhouse: {', '.join(conditions)}. "
            f"Plant health: {self._plant_health:.1%}, "
            f"Growth: {self._growth_progress:.1%}. "
            f"Step energy: {energy_step:.2f}kWh (${cost_step:.3f}). "
            f"Reward: {reward:.3f}."
        )

        if self._plant_health < 0.5:
            msg += " ⚠️ PLANT HEALTH CRITICAL!"
        if self._temperature > 35 or self._temperature < 5:
            msg += " ⚠️ EXTREME TEMPERATURE!"

        return msg


# ─── Direct testing ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Greenhouse Environment...")

    for task_id in TASK_CONFIGS:
        env = GreenhouseEnvironment(task_id=task_id)
        obs = env.reset()
        config = TASK_CONFIGS[task_id]
        print(f"\n{'='*70}")
        print(f"Task: {task_id} ({config['difficulty']}) — {config['max_steps']} steps")
        print(f"Initial: T={obs.temperature}°C, H={obs.humidity}%, "
              f"CO₂={obs.co2_level}ppm, Light={obs.light_intensity}")

        total_reward = 0
        for step in range(config["max_steps"]):
            # Simple heuristic action
            action = GreenhouseAction(
                heater_power=0.5 if obs.temperature < 22 else 0.0,
                ventilation_rate=0.3 if obs.temperature > 25 else 0.1,
                humidifier_level=0.4 if obs.humidity < 65 else 0.0,
                artificial_lighting=0.6 if obs.light_intensity < 400 and 6 <= obs.hour_of_day <= 18 else 0.0,
            )
            obs = env.step(action)
            total_reward += obs.reward

            if obs.done:
                break

        grader = obs.metadata.get("grader_score", 0) if obs.metadata else 0
        print(f"Final: T={obs.temperature}°C, Health={obs.plant_health:.2f}, "
              f"Growth={obs.growth_progress:.2f}")
        print(f"Total reward: {total_reward:.3f}, Grader score: {grader}")

    print(f"\n{'='*70}")
    print("All tasks completed successfully!")
