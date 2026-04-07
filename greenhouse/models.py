# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Greenhouse Climate Control Environment.

Defines typed Pydantic models for actions (control decisions),
observations (sensor readings + crop status), and extended state.
"""

from typing import Dict, Any, Optional, List
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class GreenhouseAction(Action):
    """
    Action for the Greenhouse Climate Control environment.

    The agent controls 4 continuous actuators to manage greenhouse climate.
    All values are normalized to [0.0, 1.0].
    """

    heater_power: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Heater power level (0.0=off, 1.0=full power). "
                    "Controls heating to raise greenhouse temperature.",
    )
    ventilation_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Ventilation fan speed (0.0=closed, 1.0=fully open). "
                    "Exchanges indoor air with outdoor air, affecting temperature and humidity.",
    )
    humidifier_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Humidifier output level (0.0=off, 1.0=full output). "
                    "Increases indoor humidity.",
    )
    artificial_lighting: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Artificial grow light intensity (0.0=off, 1.0=full brightness). "
                    "Supplements natural sunlight for photosynthesis.",
    )


class GreenhouseObservation(Observation):
    """
    Observation from the Greenhouse Climate Control environment.

    Provides the agent with all sensor readings, crop status,
    energy usage, and a human-readable status message.
    """

    # --- Indoor Climate Sensors ---
    temperature: float = Field(
        default=22.0,
        description="Indoor greenhouse temperature in °C",
    )
    humidity: float = Field(
        default=65.0,
        description="Indoor relative humidity in percent (0-100)",
    )
    co2_level: float = Field(
        default=800.0,
        description="Indoor CO₂ concentration in ppm",
    )
    light_intensity: float = Field(
        default=0.0,
        description="Total light intensity at canopy level in µmol/m²/s (PAR)",
    )

    # --- Outdoor Weather ---
    outside_temperature: float = Field(
        default=15.0,
        description="Outdoor temperature in °C",
    )
    outside_humidity: float = Field(
        default=50.0,
        description="Outdoor relative humidity in percent (0-100)",
    )
    cloud_cover: float = Field(
        default=0.3,
        description="Cloud cover fraction (0.0=clear sky, 1.0=fully overcast)",
    )

    # --- Time ---
    hour_of_day: float = Field(
        default=12.0,
        description="Current hour of the simulated day (0.0-23.99)",
    )
    day_number: int = Field(
        default=1,
        description="Current day number in the episode",
    )

    # --- Crop Status ---
    plant_health: float = Field(
        default=1.0,
        description="Plant health index (0.0=dead, 1.0=perfect health)",
    )
    growth_progress: float = Field(
        default=0.0,
        description="Cumulative growth progress (0.0=seed, 1.0=fully grown)",
    )

    # --- Energy ---
    energy_consumed_step: float = Field(
        default=0.0,
        description="Energy consumed this step in kWh",
    )
    total_energy_consumed: float = Field(
        default=0.0,
        description="Total cumulative energy consumed in kWh",
    )
    energy_cost_step: float = Field(
        default=0.0,
        description="Energy cost this step in $",
    )
    total_energy_cost: float = Field(
        default=0.0,
        description="Total cumulative energy cost in $",
    )

    # --- Episode Info ---
    step_number: int = Field(
        default=0,
        description="Current step number in the episode",
    )
    max_steps: int = Field(
        default=24,
        description="Maximum steps in this episode",
    )
    task_id: str = Field(
        default="maintain_temperature",
        description="Active task identifier",
    )

    # --- Human-readable Status ---
    status_message: str = Field(
        default="",
        description="Human-readable status summary for the LLM agent",
    )

    # --- Previous Action Echo ---
    last_action: Optional[Dict[str, float]] = Field(
        default=None,
        description="The action that was just executed (for agent reference)",
    )


class GreenhouseState(State):
    """
    Extended state for the Greenhouse environment.

    Tracks episode-level metadata beyond the base State fields.
    """

    task_id: str = Field(
        default="maintain_temperature",
        description="Active task identifier",
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward this episode",
    )
    total_energy: float = Field(
        default=0.0,
        description="Total energy consumed this episode in kWh",
    )
    plant_health: float = Field(
        default=1.0,
        description="Current plant health",
    )
    growth_progress: float = Field(
        default=0.0,
        description="Current growth progress",
    )
