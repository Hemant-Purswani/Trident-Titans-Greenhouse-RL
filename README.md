---
title: Greenhouse Climate Control Environment
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - climate-control
  - greenhouse
---

# 🌿 Greenhouse Climate Control Environment

### 👥 Team Members
- **Hemant Purswani** ([Hemant-Purswani](https://github.com/Hemant-Purswani))
- **Nikita Goswami** ([Nikieta](https://github.com/Nikieta))
- **Tirth Mehta** ([TirthM21](https://github.com/TirthM21))

A **production-grade OpenEnv environment** that simulates realistic greenhouse climate management. An AI agent controls heating, ventilation, humidity, and lighting to **optimize crop growth while minimizing energy consumption**.

This environment models a genuine real-world challenge: precision agriculture through automated greenhouse climate control — a domain where RL and LLM agents can provide immediate, practical value.

## 🎯 Why This Matters

Commercial greenhouses spend **30-40% of operating costs on climate control energy**. Intelligent automation can:
- Reduce energy consumption by 20-30%
- Improve crop yields by maintaining optimal conditions
- Respond to weather changes faster than manual control
- Operate 24/7 without human intervention

This environment provides a realistic testbed for training and evaluating such agents.

## 🚀 Quick Start

### Using the Client

```python
from greenhouse import GreenhouseEnv, GreenhouseAction

# Connect to running server
with GreenhouseEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.status_message)

    # Control the greenhouse
    action = GreenhouseAction(
        heater_power=0.5,        # 50% heater
        ventilation_rate=0.2,    # 20% ventilation
        humidifier_level=0.3,    # 30% humidifier
        artificial_lighting=0.0, # lights off
    )
    result = env.step(action)
    print(f"Temperature: {result.observation.temperature}°C")
    print(f"Plant Health: {result.observation.plant_health:.1%}")
    print(f"Reward: {result.reward:.3f}")
```

### Running Locally

```bash
# Install dependencies
uv sync

# Start the server
uv run server

# Or with uvicorn directly
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t greenhouse-env:latest -f server/Dockerfile .

# Run
docker run -d -p 8000:8000 greenhouse-env:latest

# Health check
curl http://localhost:8000/health
```

## 🎮 Tasks

The environment includes **3 progressively difficult tasks** with programmatic graders:

### Task 1: `maintain_temperature` (Easy)
- **Duration**: 24 steps (1 simulated day)
- **Goal**: Keep greenhouse temperature within 20-26°C optimal range
- **Grading**: Score = fraction of steps where temperature was in range
- **Challenge**: Day/night temperature swings, moderate weather

### Task 2: `optimize_growth` (Medium)
- **Duration**: 72 steps (3 simulated days)
- **Goal**: Maximize crop growth while minimizing energy consumption
- **Grading**: Score = growth_progress × plant_health × (1 - energy_penalty)
- **Challenge**: Balancing all climate factors, managing energy budget

### Task 3: `weather_resilience` (Hard)
- **Duration**: 168 steps (7 simulated days)
- **Goal**: Maintain optimal conditions through extreme weather events
- **Grading**: Score = 0.30×health + 0.25×growth + 0.25×avg_reward + 0.20×survival
- **Challenge**: Heat waves, cold snaps, rapid weather shifts, long-horizon planning

All grader scores are in the **0.0 – 1.0** range.

## 📤 Action Space

The agent controls 4 continuous actuators each timestep (1 step = 1 hour):

| Actuator | Range | Description |
|----------|-------|-------------|
| `heater_power` | 0.0 – 1.0 | Heating system power level |
| `ventilation_rate` | 0.0 – 1.0 | Fan speed / air exchange rate |
| `humidifier_level` | 0.0 – 1.0 | Moisture injection rate |
| `artificial_lighting` | 0.0 – 1.0 | Grow light intensity |

### Energy Costs per Step (at full power)
| Actuator | kWh | Cost |
|----------|-----|------|
| Heater | 2.0 | $0.30 |
| Ventilation | 0.3 | $0.045 |
| Humidifier | 0.2 | $0.03 |
| Lighting | 1.5 | $0.225 |

## 📥 Observation Space

Each observation includes:

### Indoor Climate (with sensor noise)
| Field | Unit | Description |
|-------|------|-------------|
| `temperature` | °C | Indoor greenhouse temperature |
| `humidity` | % | Relative humidity |
| `co2_level` | ppm | CO₂ concentration |
| `light_intensity` | µmol/m²/s | PAR at canopy level |

### Outdoor Weather
| Field | Unit | Description |
|-------|------|-------------|
| `outside_temperature` | °C | Outdoor air temperature |
| `outside_humidity` | % | Outdoor relative humidity |
| `cloud_cover` | 0.0-1.0 | Sky cloud fraction |

### Crop Status
| Field | Range | Description |
|-------|-------|-------------|
| `plant_health` | 0.0-1.0 | Health index (0=dead, 1=perfect) |
| `growth_progress` | 0.0-1.0 | Cumulative growth (0=seed, 1=mature) |

### Time & Episode
| Field | Description |
|-------|-------------|
| `hour_of_day` | 0.0 – 23.99 |
| `day_number` | Current simulated day |
| `step_number` | Current step in episode |
| `max_steps` | Total steps this task |

### Meta
| Field | Description |
|-------|-------------|
| `status_message` | Human-readable summary for LLM agents |
| `last_action` | Echo of previous action taken |
| `energy_consumed_step` | kWh consumed this step |
| `total_energy_cost` | Cumulative $ spent |

## 🌱 Optimal Growing Conditions

| Parameter | Optimal Range | Survivable Range |
|-----------|---------------|------------------|
| Temperature | 20 – 26°C | 10 – 38°C |
| Humidity | 60 – 80% | 30 – 95% |
| CO₂ | 800 – 1200 ppm | 300 – 2000 ppm |
| Light (daytime) | 400 – 800 µmol/m²/s | 50 – 1200 µmol/m²/s |

## 🏆 Reward Function

Multi-objective reward computed each step, normalized to **[0.0, 1.0]**:

```
reward = w_temp × temp_score
       + w_humid × humidity_score
       + w_light × light_score
       + w_co2 × co2_score
       + w_energy × energy_efficiency
       + w_stability × stability_bonus
```

Each score uses a linear interpolation between optimal (1.0) and survivable (0.0) ranges. Weights are task-specific to emphasize the relevant challenge.

## 🔬 Physics Simulation

The environment implements realistic greenhouse physics:

- **Thermal dynamics**: Heater input, outdoor conduction, ventilation exchange, solar radiative heating, thermal inertia
- **Humidity model**: Humidifier input, ventilation exchange, temperature-humidity coupling, plant evapotranspiration
- **CO₂ dynamics**: Natural leakage, plant photosynthesis uptake (light-dependent), soil respiration
- **Light model**: Sinusoidal day/night cycle (sunrise 6:00, sunset 18:00), cloud attenuation, artificial supplements
- **Weather system**: Sinusoidal daily temperature cycle, random walk weather trends, cloud cover Markov process, extreme weather events (hard task)
- **Sensor noise**: Gaussian noise on all sensor readings

## 📁 Project Structure

```
greenhouse/
├── __init__.py                    # Module exports
├── models.py                      # Pydantic Action/Observation/State models
├── client.py                      # GreenhouseEnv WebSocket client
├── openenv.yaml                   # OpenEnv manifest + task definitions
├── pyproject.toml                 # Dependencies and build config
├── inference.py                   # Baseline LLM inference script
├── README.md                      # This file
└── server/
    ├── __init__.py                # Server module exports
    ├── greenhouse_environment.py  # Core physics simulation + graders
    ├── app.py                     # FastAPI server (HTTP + WebSocket)
    ├── Dockerfile                 # Container image
    └── requirements.txt           # Server dependencies
```

## 🏃 Running the Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="your-token-here"

# Run baseline
python inference.py
```

The script runs all 3 tasks and outputs structured logs:
```
[START] task=maintain_temperature env=greenhouse model=meta-llama/Llama-3.3-70B-Instruct
[STEP] step=1 action={"heater_power":0.5,...} reward=0.85 done=false error=null
...
[END] success=true steps=24
```

## 🖥️ Visualization & Monitoring

The environment includes a built-in interactive dashboard for monitoring greenhouse climate trends and testing agent performance visually.

### Streamlit Dashboard

Run the dashboard after starting the server:

```bash
# 1. Start server (if not running)
python -m greenhouse.server.app

# 2. Run dashboard
streamlit run dashboard.py
```

Features:
- **Real-time Monitoring**: Visualize temperature, humidity, CO₂, and light intensity.
- **Plant Status**: Track growth progress and health metrics in real-time.
- **Manual Control**: Test the environment manually by adjusting actuator sliders.
- **Comparative Trends**: Dual-axis charts showing indoor vs. outdoor conditions.

---

## 🤖 RL Baseline Training

While the project is optimized for LLM agents via the OpenEnv spec, it remains fully compatible with traditional Reinforcement Learning (RL) frameworks.

### Stable-Baselines3 PPO

We provide a baseline training script using PPO to demonstrate the environment's utility for model-free RL.

```bash
# Train a PPO model for 50,000 steps on 'maintain_temperature' task
python train_ppo.py --task maintain_temperature --steps 50000 --output ppo_model
```

The script:
- Wraps the physics engine in an OpenAI Gym-compatible interface.
- Normalizes observations for faster convergence.
- Saves the best-performing model based on periodic evaluations.
- Logs training metrics for TensorBoard visualization.

---

## 🚀 Deployment

The project is designed for immediate deployment to **Hugging Face Spaces**.

```bash
# From the greenhouse directory
openenv push --repo-id username/greenhouse-climate-control
```

Your environment will be live at:
- **Web UI**: `https://username-greenhouse-climate-control.hf.space/web`
- **API Docs**: `https://username-greenhouse-climate-control.hf.space/docs`
- **Health**: `https://username-greenhouse-climate-control.hf.space/health`

## 📊 Baseline Scores

| Task | Difficulty | Steps | Baseline Score | Strategy |
|------|-----------|-------|----------------|----------|
| maintain_temperature | Easy | 24 | ~0.75 | Simple thermostat heuristic |
| optimize_growth | Medium | 72 | ~0.55 | Balanced multi-objective |
| weather_resilience | Hard | 168 | ~0.40 | Reactive weather response |

*Scores are approximate and will vary with the LLM model used.*

## 📄 License

BSD 3-Clause License
