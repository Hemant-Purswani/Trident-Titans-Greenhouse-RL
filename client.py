# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Greenhouse Climate Control Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import GreenhouseAction, GreenhouseObservation, GreenhouseState


class GreenhouseEnv(
    EnvClient[GreenhouseAction, GreenhouseObservation, GreenhouseState]
):
    """
    Client for the Greenhouse Climate Control Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with GreenhouseEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.status_message)
        ...
        ...     action = GreenhouseAction(
        ...         heater_power=0.5,
        ...         ventilation_rate=0.2,
        ...         humidifier_level=0.3,
        ...         artificial_lighting=0.4,
        ...     )
        ...     result = client.step(action)
        ...     print(f"Temp: {result.observation.temperature}°C")

    Example with Docker:
        >>> client = GreenhouseEnv.from_docker_image("greenhouse-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(GreenhouseAction(heater_power=0.8))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: GreenhouseAction) -> Dict:
        """
        Convert GreenhouseAction to JSON payload for step message.

        Args:
            action: GreenhouseAction with 4 control values

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "heater_power": action.heater_power,
            "ventilation_rate": action.ventilation_rate,
            "humidifier_level": action.humidifier_level,
            "artificial_lighting": action.artificial_lighting,
        }

    def _parse_result(self, payload: Dict) -> StepResult[GreenhouseObservation]:
        """
        Parse server response into StepResult[GreenhouseObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with GreenhouseObservation
        """
        obs_data = payload.get("observation", {})

        observation = GreenhouseObservation(
            # Indoor climate
            temperature=obs_data.get("temperature", 22.0),
            humidity=obs_data.get("humidity", 65.0),
            co2_level=obs_data.get("co2_level", 800.0),
            light_intensity=obs_data.get("light_intensity", 0.0),
            # Outdoor weather
            outside_temperature=obs_data.get("outside_temperature", 15.0),
            outside_humidity=obs_data.get("outside_humidity", 50.0),
            cloud_cover=obs_data.get("cloud_cover", 0.3),
            # Time
            hour_of_day=obs_data.get("hour_of_day", 12.0),
            day_number=obs_data.get("day_number", 1),
            # Crop
            plant_health=obs_data.get("plant_health", 1.0),
            growth_progress=obs_data.get("growth_progress", 0.0),
            # Energy
            energy_consumed_step=obs_data.get("energy_consumed_step", 0.0),
            total_energy_consumed=obs_data.get("total_energy_consumed", 0.0),
            energy_cost_step=obs_data.get("energy_cost_step", 0.0),
            total_energy_cost=obs_data.get("total_energy_cost", 0.0),
            # Episode
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 24),
            task_id=obs_data.get("task_id", "maintain_temperature"),
            # Status
            status_message=obs_data.get("status_message", ""),
            last_action=obs_data.get("last_action"),
            # Standard
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> GreenhouseState:
        """
        Parse server response into GreenhouseState object.

        Args:
            payload: JSON response from state request

        Returns:
            GreenhouseState with episode tracking data
        """
        return GreenhouseState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "maintain_temperature"),
            total_reward=payload.get("total_reward", 0.0),
            total_energy=payload.get("total_energy", 0.0),
            plant_health=payload.get("plant_health", 1.0),
            growth_progress=payload.get("growth_progress", 0.0),
        )
