OpenEnv: Production RL Made Simple
PyTorch
From "Hello World" to RL Training in 5 Minutes ✨
What if RL environments were as easy to use as REST APIs?

That's OpenEnv. Type-safe. Isolated. Production-ready. 🎯

Open In Colab GitHub License PyTorch

Author: Sanyam Bhutani

Why OpenEnv?
Let's take a trip down memory lane:

It's 2016, RL is popular. You read some papers, it looks promising.

But in real world: Cartpole is the best you can run on a gaming GPU.

What do you do beyond Cartpole?

Fast-forward to 2025, GRPO is awesome and this time it's not JUST in theory, it works well in practise and is really here!

The problem still remains, how do you take these RL algorithms and take them beyond Cartpole?

A huge part of RL is giving your algorithms environment access to learn.

We are excited to introduce an Environment Spec for adding Open Environments for RL Training. This will allow you to focus on your experiments and allow everyone to bring their environments.

Focus on experiments, use OpenEnvironments, and build agents that go beyond Cartpole on a single spec.

📋 What You'll Learn
🎯 Part 1-2: The Fundamentals

⚡ RL in 60 seconds
🤔 Why existing solutions fall short
💡 The OpenEnv solution
🏗️ Part 3-5: The Architecture

🔧 How OpenEnv works
🔍 Exploring real code
🎮 OpenSpiel integration example
🎮 Part 6-8: Hands-On Demo

🔌 Use existing OpenSpiel environment
🤖 Test 4 different policies
👀 Watch learning happen live
🔧 Part 9-10: Going Further

🎮 Switch to other OpenSpiel games
✨ Build your own integration
🌐 Deploy to production
!!! tip "Pro Tip" This notebook is designed to run top-to-bottom in Google Colab with zero setup!

⏱️ **Time**: ~5 minutes | 📊 **Difficulty**: Beginner-friendly | 🎯 **Outcome**: Production-ready RL knowledge
📑 Table of Contents
Foundation
Part 1: RL in 60 Seconds ⏱️
Part 2: The Problem with Traditional RL 😤
Part 3: Setup 🛠️
Architecture
Part 4: The OpenEnv Pattern 🏗️
Part 5: Example Integration - OpenSpiel 🎮
Hands-On Demo
Part 6: Interactive Demo 🎮
Part 7: Four Policies 🤖
Part 8: Policy Competition! 🏆
Advanced
Part 9: Using Real OpenSpiel 🎮
Part 10: Create Your Own Integration 🛠️
Wrap Up
Summary: Your Journey 🎓
Resources 📚
Part 1: RL in 60 Seconds ⏱️
Reinforcement Learning is simpler than you think.

It's just a loop:

while not done:
    observation = environment.observe()
    action = policy.choose(observation)
    reward = environment.step(action)
    policy.learn(reward)
That's it. That's RL.

Let's see it in action:

import random

print("🎲 " + "="*58 + " 🎲")
print("   Number Guessing Game - The Simplest RL Example")
print("🎲 " + "="*58 + " 🎲")

# Environment setup
target = random.randint(1, 10)
guesses_left = 3

print(f"\n🎯 I'm thinking of a number between 1 and 10...")
print(f"💭 You have {guesses_left} guesses. Let's see how random guessing works!\n")

# The RL Loop - Pure random policy (no learning!)
while guesses_left > 0:
    # Policy: Random guessing (no learning yet!)
    guess = random.randint(1, 10)
    guesses_left -= 1
    
    print(f"💭 Guess #{3-guesses_left}: {guess}", end=" → ")
    
    # Reward signal (but we're not using it!)
    if guess == target:
        print("🎉 Correct! +10 points")
        break
    elif abs(guess - target) <= 2:
        print("🔥 Warm! (close)")
    else:
        print("❄️  Cold! (far)")
else:
    print(f"\n💔 Out of guesses. The number was {target}.")

print("\n" + "="*62)
print("💡 This is RL: Observe → Act → Reward → Repeat")
print("   But this policy is terrible! It doesn't learn from rewards.")
print("="*62 + "\n")
Output:

🎲 ========================================================== 🎲
   Number Guessing Game - The Simplest RL Example
🎲 ========================================================== 🎲

🎯 I'm thinking of a number between 1 and 10...
💭 You have 3 guesses. Let's see how random guessing works!

💭 Guess #1: 2 → ❄️  Cold! (far)
💭 Guess #2: 10 → 🎉 Correct! +10 points

==============================================================
💡 This is RL: Observe → Act → Reward → Repeat
   But this policy is terrible! It doesn't learn from rewards.
==============================================================
Part 2: The Problem with Traditional RL 😤
🤔 Why Can't We Just Use OpenAI Gym?
Good question! Gym is great for research, but production needs more...

Challenge	Traditional Approach	OpenEnv Solution
Type Safety	❌ obs[0][3] - what is this?	✅ obs.info_state - IDE knows!
Isolation	❌ Same process (can crash your training)	✅ Docker containers (fully isolated)
Deployment	❌ "Works on my machine" 🤷	✅ Same container everywhere 🐳
Scaling	❌ Hard to distribute	✅ Deploy to Kubernetes ☸️
Language	❌ Python only	✅ Any language (HTTP API) 🌐
Debugging	❌ Cryptic numpy errors	✅ Clear type errors 🐛
💡 The OpenEnv Philosophy
"RL environments should be like microservices"

Think of it like this: You don't run your database in the same process as your web server, right? Same principle!

🔒 Isolated: Run in containers (security + stability)
🌐 Standard: HTTP API, works everywhere
📦 Versioned: Docker images (reproducibility!)
🚀 Scalable: Deploy to cloud with one command
🛡️ Type-safe: Catch bugs before they happen
🔄 Portable: Works on Mac, Linux, Windows, Cloud
The Architecture
┌────────────────────────────────────────────────────────────┐
│  YOUR TRAINING CODE                                        │
│                                                            │
│  env = OpenSpielEnv(...)        ← Import the client      │
│  result = env.reset()           ← Type-safe!             │
│  result = env.step(action)      ← Type-safe!             │
│                                                            │
└─────────────────┬──────────────────────────────────────────┘
                  │
                  │  HTTP/JSON (Language-Agnostic)
                  │  POST /reset, POST /step, GET /state
                  │
┌─────────────────▼──────────────────────────────────────────┐
│  DOCKER CONTAINER                                          │
│                                                            │
│  ┌──────────────────────────────────────────────┐         │
│  │  FastAPI Server                              │         │
│  │  └─ Environment (reset, step, state)         │         │
│  │     └─ Your Game/Simulation Logic            │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  Isolated • Reproducible • Secure                          │
└────────────────────────────────────────────────────────────┘
!!! info "Key Insight" You never see HTTP details - just clean Python methods!

```python
env.reset()    # Under the hood: HTTP POST to /reset
env.step(...)  # Under the hood: HTTP POST to /step
env.state()    # Under the hood: HTTP GET to /state
```

The magic? OpenEnv handles all the plumbing. You focus on RL! ✨
Part 3: Setup 🛠️
Running in Colab? This cell will clone OpenEnv and install dependencies automatically.

Running locally? Make sure you're in the OpenEnv directory.

# Detect environment
try:
    import google.colab
    IN_COLAB = True
    print("🌐 Running in Google Colab - Perfect!")
except ImportError:
    IN_COLAB = False
    print("💻 Running locally - Nice!")

if IN_COLAB:
    print("\n📦 Cloning OpenEnv repository...")
    !git clone https://github.com/meta-pytorch/OpenEnv.git > /dev/null 2>&1
    %cd OpenEnv
    
    print("📚 Installing dependencies (this takes ~10 seconds)...")
    !pip install -q fastapi uvicorn requests
    
    import sys
    sys.path.insert(0, './src')
    print("\n✅ Setup complete! Everything is ready to go! 🎉")
else:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent / 'src'))
    print("✅ Using local OpenEnv installation")

print("\n🚀 Ready to explore OpenEnv and build amazing things!")
print("💡 Tip: Run cells top-to-bottom for the best experience.\n")
Output:

💻 Running locally - Nice!
✅ Using local OpenEnv installation

🚀 Ready to explore OpenEnv and build amazing things!
💡 Tip: Run cells top-to-bottom for the best experience.
Part 4: The OpenEnv Pattern 🏗️
Every OpenEnv Environment Has 3 Components:
src/envs/your_env/
├── 📝 models.py          ← Type-safe contracts
│                           (Action, Observation, State)
│
├── 📱 client.py          ← What YOU import
│                           (HTTPEnvClient implementation)
│
└── 🖥️  server/
    ├── environment.py    ← Game/simulation logic
    ├── app.py            ← FastAPI server
    └── Dockerfile        ← Container definition
Let's explore the actual OpenEnv code to see how this works:

# Import OpenEnv's core abstractions
from core.env_server import Environment, Action, Observation, State
from core.http_env_client import HTTPEnvClient

print("="*70)
print("   🧩 OPENENV CORE ABSTRACTIONS")
print("="*70)

print("""
🖥️  SERVER SIDE (runs in Docker):

    class Environment(ABC):
        '''Base class for all environment implementations'''
        
        @abstractmethod
        def reset(self) -> Observation:
            '''Start new episode'''
        
        @abstractmethod
        def step(self, action: Action) -> Observation:
            '''Execute action, return observation'''
        
        @property
        def state(self) -> State:
            '''Get episode metadata'''

📱 CLIENT SIDE (your training code):

    class HTTPEnvClient(ABC):
        '''Base class for HTTP clients'''
        
        def reset(self) -> StepResult:
            # HTTP POST /reset
        
        def step(self, action) -> StepResult:
            # HTTP POST /step
        
        def state(self) -> State:
            # HTTP GET /state
""")

print("="*70)
print("\n✨ Same interface on both sides - communication via HTTP!")
print("🎯 You focus on RL, OpenEnv handles the infrastructure.\n")
Output:

======================================================================
   🧩 OPENENV CORE ABSTRACTIONS
======================================================================

🖥️  SERVER SIDE (runs in Docker):

    class Environment(ABC):
        '''Base class for all environment implementations'''
        
        @abstractmethod
        def reset(self) -> Observation:
            '''Start new episode'''
        
        @abstractmethod
        def step(self, action: Action) -> Observation:
            '''Execute action, return observation'''
        
        @property
        def state(self) -> State:
            '''Get episode metadata'''

📱 CLIENT SIDE (your training code):

    class HTTPEnvClient(ABC):
        '''Base class for HTTP clients'''
        
        def reset(self) -> StepResult:
            # HTTP POST /reset
        
        def step(self, action) -> StepResult:
            # HTTP POST /step
        
        def state(self) -> State:
            # HTTP GET /state

======================================================================

✨ Same interface on both sides - communication via HTTP!
🎯 You focus on RL, OpenEnv handles the infrastructure.
Part 5: Example Integration - OpenSpiel 🎮
What is OpenSpiel?
OpenSpiel is a library from DeepMind with 70+ game environments for RL research.

OpenEnv's Integration
We've wrapped 6 OpenSpiel games following the OpenEnv pattern:

🎯 Single-Player	👥 Multi-Player
1. Catch - Catch falling ball	5. Tic-Tac-Toe - Classic 3×3
2. Cliff Walking - Navigate grid	6. Kuhn Poker - Imperfect info poker
3. 2048 - Tile puzzle	
4. Blackjack - Card game	
This shows how OpenEnv can wrap any existing RL library!

from envs.openspiel_env.client import OpenSpielEnv

print("="*70)
print("   🔌 HOW OPENENV WRAPS OPENSPIEL")
print("="*70)

print("""
class OpenSpielEnv(HTTPEnvClient[OpenSpielAction, OpenSpielObservation]):
    
    def _step_payload(self, action: OpenSpielAction) -> dict:
        '''Convert typed action to JSON for HTTP'''
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
        }
    
    def _parse_result(self, payload: dict) -> StepResult:
        '''Parse HTTP JSON response into typed observation'''
        return StepResult(
            observation=OpenSpielObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )

""")

print("─" * 70)
print("\n✨ Usage (works for ALL OpenEnv environments):")
print("""
  env = OpenSpielEnv(base_url="http://localhost:8000")
  
  result = env.reset()
  # Returns StepResult[OpenSpielObservation] - Type safe!
  
  result = env.step(OpenSpielAction(action_id=2, game_name="catch"))
  # Type checker knows this is valid!
  
  state = env.state()
  # Returns OpenSpielState
""")

print("─" * 70)
print("\n🎯 This pattern works for ANY environment you want to wrap!\n")
Output:

======================================================================
   🔌 HOW OPENENV WRAPS OPENSPIEL
======================================================================

class OpenSpielEnv(HTTPEnvClient[OpenSpielAction, OpenSpielObservation]):
    
    def _step_payload(self, action: OpenSpielAction) -> dict:
        '''Convert typed action to JSON for HTTP'''
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
        }
    
    def _parse_result(self, payload: dict) -> StepResult:
        '''Parse HTTP JSON response into typed observation'''
        return StepResult(
            observation=OpenSpielObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )


──────────────────────────────────────────────────────────────────────

✨ Usage (works for ALL OpenEnv environments):

  env = OpenSpielEnv(base_url="http://localhost:8000")
  
  result = env.reset()
  # Returns StepResult[OpenSpielObservation] - Type safe!
  
  result = env.step(OpenSpielAction(action_id=2, game_name="catch"))
  # Type checker knows this is valid!
  
  state = env.state()
  # Returns OpenSpielState

──────────────────────────────────────────────────────────────────────

🎯 This pattern works for ANY environment you want to wrap!
Type-Safe Models
# Import OpenSpiel integration models
from envs.openspiel_env.models import (
    OpenSpielAction,
    OpenSpielObservation,
    OpenSpielState
)
from dataclasses import fields

print("="*70)
print("   🎮 OPENSPIEL INTEGRATION - TYPE-SAFE MODELS")
print("="*70)

print("\n📤 OpenSpielAction (what you send):")
print("   " + "─" * 64)
for field in fields(OpenSpielAction):
    print(f"   • {field.name:20s} : {field.type}")

print("\n📥 OpenSpielObservation (what you receive):")
print("   " + "─" * 64)
for field in fields(OpenSpielObservation):
    print(f"   • {field.name:20s} : {field.type}")

print("\n📊 OpenSpielState (episode metadata):")
print("   " + "─" * 64)
for field in fields(OpenSpielState):
    print(f"   • {field.name:20s} : {field.type}")

print("\n" + "="*70)
print("\n💡 Type safety means:")
print("   ✅ Your IDE autocompletes these fields")
print("   ✅ Typos are caught before running")
print("   ✅ Refactoring is safe")
print("   ✅ Self-documenting code\n")
Output:

======================================================================
   🎮 OPENSPIEL INTEGRATION - TYPE-SAFE MODELS
======================================================================

📤 OpenSpielAction (what you send):
   ────────────────────────────────────────────────────────────────
   • metadata             : typing.Dict[str, typing.Any]
   • action_id            : int
   • game_name            : str
   • game_params          : Dict[str, Any]

📥 OpenSpielObservation (what you receive):
   ────────────────────────────────────────────────────────────────
   • done                 : <class 'bool'>
   • reward               : typing.Union[bool, int, float, NoneType]
   • metadata             : typing.Dict[str, typing.Any]
   • info_state           : List[float]
   • legal_actions        : List[int]
   • game_phase           : str
   • current_player_id    : int
   • opponent_last_action : Optional[int]

📊 OpenSpielState (episode metadata):
   ────────────────────────────────────────────────────────────────
   • episode_id           : typing.Optional[str]
   • step_count           : <class 'int'>
   • game_name            : str
   • agent_player         : int
   • opponent_policy      : str
   • game_params          : Dict[str, Any]
   • num_players          : int

======================================================================

💡 Type safety means:
   ✅ Your IDE autocompletes these fields
   ✅ Typos are caught before running
   ✅ Refactoring is safe
   ✅ Self-documenting code
How the Client Works
The client inherits from HTTPEnvClient and implements 3 methods:

_step_payload() - Convert action → JSON
_parse_result() - Parse JSON → typed observation
_parse_state() - Parse JSON → state
That's it! The base class handles all HTTP communication.

Part 6: Using Real OpenSpiel 🎮
Now let's USE a production environment!
We'll play Catch using OpenEnv's OpenSpiel integration 🎯

This is a REAL environment running in production at companies!

Get ready for:

🔌 Using existing environments (not building)
🤖 Testing policies against real games
📊 Live gameplay visualization
🎯 Production-ready patterns
The Game: Catch 🔴🏓
⬜ ⬜ 🔴 ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜   Ball
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜   falls
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜   down
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ ⬜ ⬜ ⬜
⬜ ⬜ 🏓 ⬜ ⬜
     Paddle
Rules:

10×5 grid
Ball falls from random column
Move paddle left/right to catch it
Actions:

0 = Move LEFT ⬅️
1 = STAY 🛑
2 = Move RIGHT ➡️
Reward:

+1 if caught 🎉
0 if missed 😢
!!! note "Why Catch?" - Simple rules (easy to understand) - Fast episodes (~5 steps) - Clear success/failure - Part of OpenSpiel's 70+ games!

**💡 The Big Idea:**
Instead of building this from scratch, we'll USE OpenEnv's existing OpenSpiel integration. Same interface, but production-ready!
from envs.openspiel_env import OpenSpielEnv
from envs.openspiel_env.models import (
    OpenSpielAction,
    OpenSpielObservation,
    OpenSpielState
)
from dataclasses import fields

print("🎮 " + "="*64 + " 🎮")
print("   ✅ Importing Real OpenSpiel Environment!")
print("🎮 " + "="*64 + " 🎮\n")

print("📦 What we just imported:")
print("   • OpenSpielEnv - HTTP client for OpenSpiel games")
print("   • OpenSpielAction - Type-safe actions")
print("   • OpenSpielObservation - Type-safe observations")
print("   • OpenSpielState - Episode metadata\n")

print("📋 OpenSpielObservation fields:")
print("   " + "─" * 60)
for field in fields(OpenSpielObservation):
    print(f"   • {field.name:25s} : {field.type}")

print("\n" + "="*70)
print("\n💡 This is REAL OpenEnv code - used in production!")
print("   • Wraps 6 OpenSpiel games (Catch, Tic-Tac-Toe, Poker, etc.)")
print("   • Type-safe actions and observations")
print("   • Works via HTTP (we'll see that next!)\n")
Output:

🎮 ================================================================ 🎮
   ✅ Importing Real OpenSpiel Environment!
🎮 ================================================================ 🎮

📦 What we just imported:
   • OpenSpielEnv - HTTP client for OpenSpiel games
   • OpenSpielAction - Type-safe actions
   • OpenSpielObservation - Type-safe observations
   • OpenSpielState - Episode metadata

📋 OpenSpielObservation fields:
   ────────────────────────────────────────────────────────────
   • done                      : <class 'bool'>
   • reward                    : typing.Union[bool, int, float, NoneType]
   • metadata                  : typing.Dict[str, typing.Any]
   • info_state                : List[float]
   • legal_actions             : List[int]
   • game_phase                : str
   • current_player_id         : int
   • opponent_last_action      : Optional[int]

======================================================================

💡 This is REAL OpenEnv code - used in production!
   • Wraps 6 OpenSpiel games (Catch, Tic-Tac-Toe, Poker, etc.)
   • Type-safe actions and observations
   • Works via HTTP (we'll see that next!)
Part 7: Four Policies 🤖
Let's test 4 different AI strategies:

Policy	Strategy	Expected Performance
🎲 Random	Pick random action every step	~20% (pure luck)
🛑 Always Stay	Never move, hope ball lands in center	~20% (terrible!)
🧠 Smart	Move paddle toward ball	100% (optimal!)
📈 Learning	Start random, learn smart strategy	~85% (improves over time)
💡 These policies work with ANY OpenSpiel game!

import random

# ============================================================================
# POLICIES - Different AI strategies (adapted for OpenSpiel)
# ============================================================================

class RandomPolicy:
    """Baseline: Pure random guessing."""
    name = "🎲 Random Guesser"

    def select_action(self, obs: OpenSpielObservation) -> int:
        return random.choice(obs.legal_actions)


class AlwaysStayPolicy:
    """Bad strategy: Never moves."""
    name = "🛑 Always Stay"

    def select_action(self, obs: OpenSpielObservation) -> int:
        return 1  # STAY


class SmartPolicy:
    """Optimal: Move paddle toward ball."""
    name = "🧠 Smart Heuristic"

    def select_action(self, obs: OpenSpielObservation) -> int:
        # Parse OpenSpiel observation
        # For Catch: info_state is a flattened 10x5 grid
        # Ball position and paddle position encoded in the vector
        info_state = obs.info_state

        # Find ball and paddle positions from info_state
        # Catch uses a 10x5 grid, so 50 values
        grid_size = 5

        # Find positions (ball = 1.0 in the flattened grid, paddle = 1.0 in the last row of the flattened grid)
        ball_col = None
        paddle_col = None

        for idx, val in enumerate(info_state):
            if abs(val - 1.0) < 0.01:  # Ball
                ball_col = idx % grid_size
                break

        last_row = info_state[-grid_size:]
        paddle_col = last_row.index(1.0) # Paddle

        if ball_col is not None and paddle_col is not None:
            if paddle_col < ball_col:
                return 2  # Move RIGHT
            elif paddle_col > ball_col:
                return 0  # Move LEFT

        return 1  # STAY (fallback)


class LearningPolicy:
    """Simulated RL: Epsilon-greedy exploration."""
    name = "📈 Learning Agent"

    def __init__(self):
        self.steps = 0
        self.smart_policy = SmartPolicy()

    def select_action(self, obs: OpenSpielObservation) -> int:
        self.steps += 1

        # Decay exploration rate over time
        epsilon = max(0.1, 1.0 - (self.steps / 100))

        if random.random() < epsilon:
            # Explore: random action
            return random.choice(obs.legal_actions)
        else:
            # Exploit: use smart strategy
            return self.smart_policy.select_action(obs)


print("🤖 " + "="*64 + " 🤖")
print("   ✅ 4 Policies Created (Adapted for OpenSpiel)!")
print("🤖 " + "="*64 + " 🤖\n")

policies = [RandomPolicy(), AlwaysStayPolicy(), SmartPolicy(), LearningPolicy()]
for i, policy in enumerate(policies, 1):
    print(f"   {i}. {policy.name}")

print("\n💡 These policies work with OpenSpielObservation!")
print("   • Read info_state (flattened grid)")
print("   • Use legal_actions")
print("   • Work with ANY OpenSpiel game that exposes these!\n")
Output:

🤖 ================================================================ 🤖
   ✅ 4 Policies Created (Adapted for OpenSpiel)!
🤖 ================================================================ 🤖

   1. 🎲 Random Guesser
   2. 🛑 Always Stay
   3. 🧠 Smart Heuristic
   4. 📈 Learning Agent

💡 These policies work with OpenSpielObservation!
   • Read info_state (flattened grid)
   • Use legal_actions
   • Work with ANY OpenSpiel game that exposes these!
Part 8: Policy Competition! 🏆
Let's run 50 episodes for each policy against REAL OpenSpiel and see who wins!

This is production code - every action is an HTTP call to the OpenSpiel server!

def evaluate_policies(env, num_episodes=50):
    """Compare all policies over many episodes using real OpenSpiel."""
    policies = [
        RandomPolicy(),
        AlwaysStayPolicy(),
        SmartPolicy(),
        LearningPolicy(),
    ]

    print("\n🏆 " + "="*66 + " 🏆")
    print(f"   POLICY SHOWDOWN - {num_episodes} Episodes Each")
    print(f"   Playing against REAL OpenSpiel Catch!")
    print("🏆 " + "="*66 + " 🏆\n")

    results = []
    for policy in policies:
        print(f"⚡ Testing {policy.name}...", end=" ")
        successes = sum(run_episode(env, policy, visualize=False)
                       for _ in range(num_episodes))
        success_rate = (successes / num_episodes) * 100
        results.append((policy.name, success_rate, successes))
        print(f"✓ Done!")

    print("\n" + "="*70)
    print("   📊 FINAL RESULTS")
    print("="*70 + "\n")

    # Sort by success rate (descending)
    results.sort(key=lambda x: x[1], reverse=True)

    # Award medals to top 3
    medals = ["🥇", "🥈", "🥉", "  "]

    for i, (name, rate, successes) in enumerate(results):
        medal = medals[i]
        bar = "█" * int(rate / 2)
        print(f"{medal} {name:25s} [{bar:<50}] {rate:5.1f}% ({successes}/{num_episodes})")

    print("\n" + "="*70)
    print("\n✨ Key Insights:")
    print("   • Random (~20%):      Baseline - pure luck 🎲")
    print("   • Always Stay (~20%): Bad strategy - stays center 🛑")
    print("   • Smart (100%):       Optimal - perfect play! 🧠")
    print("   • Learning (~85%):    Improves over time 📈")
    print("\n🎓 This is Reinforcement Learning + OpenEnv in action:")
    print("   1. We USED existing OpenSpiel environment (didn't build it)")
    print("   2. Type-safe communication over HTTP")
    print("   3. Same code works for ANY OpenSpiel game")
    print("   4. Production-ready architecture\n")

# Run the epic competition!
print("🎮 Starting the showdown against REAL OpenSpiel...\n")
evaluate_policies(client, num_episodes=50)
Part 9: Switching to Other Games 🎮
What We Just Used: Real OpenSpiel! 🎉
In Parts 6-8, we USED the existing OpenSpiel Catch environment:

What We Did	How It Works
Imported	OpenSpielEnv client (pre-built)
Started	OpenSpiel server via uvicorn
Connected	HTTP client to server
Played	Real OpenSpiel Catch game
🎯 This is production code! Every action was an HTTP call to a real OpenSpiel environment.

🎮 6 Games Available - Same Interface!
The beauty of OpenEnv? Same code, different games!

# We just used Catch
env = OpenSpielEnv(base_url="http://localhost:8000")
# game_name="catch" was set via environment variable

# Want Tic-Tac-Toe instead? Just change the game!
# Start server with: OPENSPIEL_GAME=tic_tac_toe uvicorn ...
# Same client code works!
🎮 All 6 Games:

✅ catch - What we just used!
tic_tac_toe - Classic 3×3
kuhn_poker - Imperfect information poker
cliff_walking - Grid navigation
2048 - Tile puzzle
blackjack - Card game
All use the exact same OpenSpielEnv client!

Try Another Game (Optional):
# Stop the current server (kill the server_process)
# Then start a new game:

server_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn",
     "envs.openspiel_env.server.app:app",
     "--host", "0.0.0.0",
     "--port", "8000"],
    env={**os.environ,
         "PYTHONPATH": f"{work_dir}/src",
         "OPENSPIEL_GAME": "tic_tac_toe",  # Changed!
         "OPENSPIEL_AGENT_PLAYER": "0",
         "OPENSPIEL_OPPONENT_POLICY": "random"},
    # ... rest of config
)

# Same client works!
client = OpenSpielEnv(base_url="http://localhost:8000")
result = client.reset()  # Now playing Tic-Tac-Toe!
💡 Key Insight: You don't rebuild anything - you just USE different games with the same client!

Part 10: Create Your Own Integration 🛠️
The 5-Step Pattern
Want to wrap your own environment in OpenEnv? Here's how:

Step 1: Define Types (models.py)
from dataclasses import dataclass
from core.env_server import Action, Observation, State

@dataclass
class YourAction(Action):
    action_value: int
    # Add your action fields

@dataclass
class YourObservation(Observation):
    state_data: List[float]
    done: bool
    reward: float
    # Add your observation fields

@dataclass
class YourState(State):
    episode_id: str
    step_count: int
    # Add your state fields
Step 2: Implement Environment (server/environment.py)
from core.env_server import Environment

class YourEnvironment(Environment):
    def reset(self) -> Observation:
        # Initialize your game/simulation
        return YourObservation(...)
    
    def step(self, action: Action) -> Observation:
        # Execute action, update state
        return YourObservation(...)
    
    @property
    def state(self) -> State:
        return self._state
Step 3: Create Client (client.py)
from core.http_env_client import HTTPEnvClient
from core.types import StepResult

class YourEnv(HTTPEnvClient[YourAction, YourObservation]):
    def _step_payload(self, action: YourAction) -> dict:
        """Convert action to JSON"""
        return {"action_value": action.action_value}
    
    def _parse_result(self, payload: dict) -> StepResult:
        """Parse JSON to observation"""
        return StepResult(
            observation=YourObservation(...),
            reward=payload['reward'],
            done=payload['done']
        )
    
    def _parse_state(self, payload: dict) -> YourState:
        return YourState(...)
Step 4: Create Server (server/app.py)
from core.env_server import create_fastapi_app
from .your_environment import YourEnvironment

env = YourEnvironment()
app = create_fastapi_app(env)

# That's it! OpenEnv creates all endpoints for you.
Step 5: Dockerize (server/Dockerfile)
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
🎓 Examples to Study
OpenEnv includes 3 complete examples:

src/envs/echo_env/

Simplest possible environment
Great for testing and learning
src/envs/openspiel_env/

Wraps external library (OpenSpiel)
Shows integration pattern
6 games in one integration
src/envs/coding_env/

Python code execution environment
Shows complex use case
Security considerations
💡 Study these to understand the patterns!

🎓 Summary: Your Journey
What You Learned
📚 Concepts
✅ RL Fundamentals

The observe-act-reward loop
What makes good policies
Exploration vs exploitation
✅ OpenEnv Architecture

Client-server separation
Type-safe contracts
HTTP communication layer
✅ Production Patterns

Docker isolation
API design
Reproducible deployments
🛠️ Skills
✅ Using Environments

Import OpenEnv clients
Call reset/step/state
Work with typed observations
✅ Building Environments

Define type-safe models
Implement Environment class
Create HTTPEnvClient
✅ Testing & Debugging

Compare policies
Visualize episodes
Measure performance
OpenEnv vs Traditional RL
Feature	Traditional (Gym)	OpenEnv	Winner
Type Safety	❌ Arrays, dicts	✅ Dataclasses	🏆 OpenEnv
Isolation	❌ Same process	✅ Docker	🏆 OpenEnv
Deployment	❌ Manual setup	✅ K8s-ready	🏆 OpenEnv
Language	❌ Python only	✅ Any (HTTP)	🏆 OpenEnv
Reproducibility	❌ "Works on my machine"	✅ Same everywhere	🏆 OpenEnv
Community	✅ Large ecosystem	🟡 Growing	🤝 Both!
!!! success "The Bottom Line" OpenEnv brings production engineering to RL:

- Same environments work locally and in production
- Type safety catches bugs early
- Docker isolation prevents conflicts
- HTTP API works with any language

**It's RL for 2024 and beyond.**
📚 Resources
🔗 Essential Links
🏠 OpenEnv GitHub: https://github.com/meta-pytorch/OpenEnv
🎮 OpenSpiel: https://github.com/google-deepmind/open_spiel
⚡ FastAPI Docs: https://fastapi.tiangolo.com/
🐳 Docker Guide: https://docs.docker.com/get-started/
🔥 PyTorch: https://pytorch.org/
📖 Documentation Deep Dives
Environment Creation Guide: src/envs/README.md
OpenSpiel Integration: src/envs/openspiel_env/README.md
Example Scripts: examples/
RFC 001: Baseline API Specs
🎓 Community & Support
Supported by amazing organizations:

🔥 Meta PyTorch
🤗 Hugging Face
⚡ Unsloth AI
🌟 Reflection AI
🚀 And many more!
License: BSD 3-Clause (very permissive!)

Contributions: Always welcome! Check out the issues tab.

🌈 What's Next?
⭐ Star the repo to show support and stay updated
🔄 Try modifying the Catch game (make it harder? bigger grid?)
🎮 Explore other OpenSpiel games
🛠️ Build your own environment integration
💬 Share what you build with the community!

2. Deploying an OpenEnv environment
This section covers deploying OpenEnv environments locally, on clusters, and on Hugging Face Spaces.

Contents:

Local Development with Uvicorn
Docker Deployment
Hugging Face Spaces
Best Practices
HF Spaces are the infrastructure for OpenEnv environments
Every HF Space provides three things that OpenEnv environments need:

Component	What it provides	How to access	Used as
Server	Running environment endpoint	https://<username>-<space-name>.hf.space	Agent and Public API
Repository	Installable Python package	pip install git+https://huggingface.co/spaces/<username>-<space-name>	Code and client
Registry	Docker container image	docker pull registry.hf.space/<username>-<space-name>:latest	Deployment
This means a single Space deployment gives you all the components you need to use an environment in training.

1. Server: A running environment endpoint
When you deploy to HF Spaces, your environment runs as a server. The client connects via WebSocket (/ws) for a persistent session:

from echo_env import EchoEnv, EchoAction

# Connect directly to the running Space (WebSocket under the hood)
# Async (recommended):
async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
    result = await client.reset()
    result = await client.step(EchoAction(message="Hello"))

# Sync (using .sync() wrapper):
with EchoEnv(base_url="https://openenv-echo-env.hf.space").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello"))
Endpoints available:

Endpoint	Protocol	Description
/ws	WebSocket	Persistent session (used by client)
/health	HTTP GET	Health check
/reset	HTTP POST	Reset environment (stateless)
/step	HTTP POST	Execute action (stateless)
/state	HTTP GET	Get current state
/docs	HTTP GET	OpenAPI documentation
/web	HTTP GET	Interactive web UI
Note: The Python client uses the /ws WebSocket endpoint by default. HTTP endpoints are available for debugging or stateless use cases.

Example: Check if a Space is running

curl https://openenv-echo-env.hf.space/health
# {"status": "healthy"}
2. Repository: Installable Python package
Every Space is a Git repository. OpenEnv environments include a pyproject.toml, making them pip-installable directly from the Space URL.

# Install client package from Space
pip install git+https://huggingface.co/spaces/openenv/echo-env
This installs:

Client class (EchoEnv) — Handles HTTP/WebSocket communication
Models (EchoAction, EchoObservation) — Typed action and observation classes
Utilities — Any helper functions the environment provides
After installation:

from envs.echo_env import EchoEnv, EchoAction, EchoObservation

# Now you have typed classes for the environment
action = EchoAction(message="Hello")
3. Registry: Docker container image
Every Docker-based Space has a container registry. You can pull and run the environment locally.

# Pull the image
docker pull registry.hf.space/openenv-echo-env:latest

# Run locally on port 8001
docker run -d -p 8001:8000 registry.hf.space/openenv-echo-env:latest
Find the registry URL for any Space:

Go to the Space page (e.g., openenv/echo-env)
Click ⋮ (three dots) → "Run locally"
Copy the docker run command
Choosing an access method
Method	Use when	Pros	Cons
Server	Quick testing, low volume	Zero setup	Network latency, rate limits
Repository	Need typed classes	Type safety, IDE support	Still need a server
Docker	Local dev, high throughput	Full control, no network	Requires Docker
Typical workflow:

import asyncio
from echo_env import EchoEnv, EchoAction

async def main():
    # Development: connect to remote Space
    async with EchoEnv(base_url="https://openenv-echo-env.hf.space") as client:
        result = await client.reset()

    # Production: run locally for speed
    # docker run -d -p 8001:8000 registry.hf.space/openenv-echo-env:latest
    async with EchoEnv(base_url="http://localhost:8001") as client:
        result = await client.reset()

    # Or let the client manage Docker for you
    client = await EchoEnv.from_env("openenv/echo-env")  # Auto-pulls and runs
    async with client:
        result = await client.reset()

asyncio.run(main())

# For sync usage, use the .sync() wrapper:
with EchoEnv(base_url="http://localhost:8001").sync() as client:
    result = client.reset()
Reference: HF Spaces Documentation | Environment Hub Collection

Local Development with Uvicorn
The fastest way to iterate on environment logic is running directly with Uvicorn.

Clone and run the environment locally
# Clone from HF Space
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark

# Install in editable mode
uv sync

# Start server
uv run server

# Run isolated from remote Space
uv run --isolated --project https://huggingface.co/spaces/burtenshaw/openenv-benchmark server
Uvicorn directly in python
# Full control over uvicorn options
uvicorn benchmark.server.app:app --host "$HOST" --port "$PORT" --workers "$WORKERS"

# With reload for development
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --reload

# Multi-Worker Mode For better concurrency:
uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --workers 4
Flag	Purpose
--reload	Auto-restart on code changes
--workers N	Run N worker processes
--log-level debug	Verbose logging
Docker Deployment
Docker provides isolation and reproducibility for production use.

Run the environment locally from the space
# Run the environment locally from the space
docker run -d -p 8000:8000 registry.hf.space/openenv-echo-env:latest
Build Image
# Clone from HF Space
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark

# Using OpenEnv CLI (recommended)
openenv build -t openenv-benchmark:latest

# Or with Docker directly
docker build -t openenv-benchmark:latest -f server/Dockerfile .
Run Container
# Basic run
docker run -d -p 8000:8000 my-env:latest

# With environment variables
docker run -d -p 8000:8000 \
    -e WORKERS=4 \
    -e MAX_CONCURRENT_ENVS=100 \
    my-env:latest

# Named container for easy management
docker run -d --name my-env -p 8000:8000 my-env:latest
Connect from Python
import asyncio
from echo_env import EchoEnv, EchoAction

async def main():
    # Async usage (recommended)
    async with EchoEnv(base_url="http://localhost:8000") as client:
        result = await client.reset()
        result = await client.step(EchoAction(message="Hello"))
        print(result.observation)

    # From Docker image
    client = await EchoEnv.from_docker_image("<local_docker_image>")
    async with client:
        result = await client.reset()
        print(result.observation)

asyncio.run(main())

# Sync usage (using .sync() wrapper)
with EchoEnv(base_url="http://localhost:8000").sync() as client:
    result = client.reset()
    result = client.step(EchoAction(message="Hello"))
    print(result.observation)
Container Lifecycle
Method	Container	WebSocket	On close()
from_hub(repo_id)	Starts	Connects	Stops container
from_hub(repo_id, use_docker=False)	None (UV)	Connects	Stops UV server
from_docker_image(image)	Starts	Connects	Stops container
MyEnv(base_url=...)	None	Connects	Disconnects only
Find Docker Commands for Any Space

Open the Space on HuggingFace Hub
Click ⋮ (three dots) menu
Select "Run locally"
Copy the provided docker run command
Deploy with CLI
cd my_env

# Deploy to your namespace
openenv push

# Deploy to specific repo
openenv push --repo-id username/my-env

# Deploy as private
openenv push --repo-id username/my-env --private
Space Configuration
The openenv.yaml manifest controls Space settings:

# openenv.yaml
name: my_env
version: "1.0.0"
description: My custom environment
Hardware Options:

Tier	vCPU	RAM	Cost
CPU Basic (Free)	2	16GB	Free
CPU Upgrade	8	32GB	$0.03/hr
OpenEnv environments support configuration via environment variables.

Variable	Default	Description
WORKERS	4	Uvicorn worker processes
PORT	8000	Server port
HOST	0.0.0.0	Bind address
MAX_CONCURRENT_ENVS	100	Max WebSocket sessions
ENABLE_WEB_INTERFACE	Auto	Enable web UI
Environment-Specific Variables
Some environments have custom variables:

TextArena:

TEXTARENA_ENV_ID=Wordle-v0
TEXTARENA_NUM_PLAYERS=1
TEXTARENA_MAX_TURNS=6
Coding Environment:

SANDBOX_TIMEOUT=30
MAX_OUTPUT_LENGTH=10000
DEMO: Deploying to Hugging Face Spaces
This demo walks through the full workflow: create an environment, test locally, deploy to HF Spaces, and use it.

Step 1: Initialize a new environment
openenv init my_env
cd my_env
This creates the standard OpenEnv structure:

my_env/
├── server/
│   ├── app.py           # FastAPI server
│   ├── environment.py   # Your environment logic
│   └── Dockerfile
├── models.py            # Action/Observation types
├── client.py            # HTTP client
├── openenv.yaml         # Manifest
└── pyproject.toml
Step 2: Run locally
# Start the server
uv run server

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
Test the health endpoint:

curl http://localhost:8000/health
# {"status": "healthy"}
Step 3: Deploy to HF Spaces
openenv push --repo-id username/my-env
Your environment is now live at:

Web UI: https://username-my-env.hf.space/web
API Docs: https://username-my-env.hf.space/docs
Health: https://username-my-env.hf.space/health
curl https://openenv-echo-env.hf.space/health
# {"status": "healthy"}
Step 4: install the environment
uv pip install git+https://huggingface.co/spaces/openenv/echo_env
Step 5: Run locally via Docker (optional)
Pull and run the container from the HF registry, or open the browser:

# Pull from HF Spaces registry
docker pull registry.hf.space/openenv-echo-env:latest

# Run locally
docker run -it -p 7860:7860 --platform=linux/amd64 \
	registry.hf.space/openenv-echo-env:latest
Now connect to your local instance:

import asyncio
from echo_env import EchoEnv, EchoAction

# Async (recommended)
async def main():
    async with EchoEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        print(result.observation)
        result = await env.step(EchoAction(message="Hello"))
        print(result.observation)

asyncio.run(main())

# Sync (using .sync() wrapper)
with EchoEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print(result.observation)
    result = env.step(EchoAction(message="Hello"))
    print(result.observation)

    3. How OpenEnv environments scale
This section covers benchmarking and scaling OpenEnv environments.

Contents:

Provider Scaling
WebSocket-based Scaling
Microservice Scaling
Scaling Experiments
Provider Scaling
The easiest way to scale an OpenEnv environment is to use a provider these are abstractions based on runtimes like Uvicorn, Docker Swarm, or Kubernetes.

from openenv.providers import UVProvider, DockerSwarmProvider, LocalDockerProvider

docker_provider = LocalDockerProvider() # default
uvicorn_provider = UVProvider() # python only
swarm_provider = DockerSwarmProvider() 

with EchoEnv.from_hub(
    repo_id="openenv/echo-env", 
    provider=swarm_provider, 
    replicas=4,
) as env:
  result = env.reset()
  result = env.step(EchoAction(message="Hello"))
WebSocket-based Scaling
OpenEnv uses WebSocket connections (/ws) instead of stateless HTTP for environment interactions. This design enables efficient scaling within a single container.

What are WebSockets?
WebSocket is a communication protocol that provides a persistent, bidirectional connection between client and server. Unlike HTTP—where each request opens a new connection, sends data, receives a response, and closes—a WebSocket connection stays open for the duration of a session.

WebSocket vs HTTP

For RL environments, this matters because a typical episode involves dozens to thousands of sequential step() calls. With HTTP, each step incurs TCP handshake overhead (~10-50ms). With WebSocket, messages are sent as lightweight frames (~0.1ms overhead) over the existing connection.

Also, with HTTP, long running sessions require logic to manage session state, which is not necessary with WebSocket.

Multiple sessions per container
With HTTP, maintaining session state requires cookies or session IDs with every request. Each isolated environment instance typically needs its own container:

HTTP approach: N parallel episodes → N containers
Note

This is completely fine (and ideal) for larger deployments where containers can be scaled. But if your resources are constrained, this add loads of overhead.

With WebSocket, one container handles many isolated sessions. Each WebSocket connection gets its own environment instance server-side:

# Single container serving multiple concurrent sessions
# docker run -d -p 8000:8000 my-env:latest

# Each client gets an isolated environment instance
with MyEnv(base_url="http://localhost:8000") as env1:  # Session 1
    result = env1.reset()
    
with MyEnv(base_url="http://localhost:8000") as env2:  # Session 2
    result = env2.reset()
    
with MyEnv(base_url="http://localhost:8000") as env3:  # Session 3
    result = env3.reset()
Note

This has its own advantages and disadvantages. For example: Separation of concerns and fault tolerance in environments like coding or terminal.

Server-side session state
The server maintains environment state per WebSocket connection which means that the environment builder does not need to worry about session state.

No session IDs because Connection itself is the session
Automatic cleanup because Environment instance destroyed when connection closes
Isolation guaranteed because Each connection has dedicated state
# Server creates new environment instance per WebSocket connection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    env = MyEnvironment()  # Fresh instance per connection
    await websocket.accept()
    
    while True:
        data = await websocket.receive_json()
        if data["type"] == "reset":
            result = env.reset()
        elif data["type"] == "step":
            result = env.step(data["action"])
        await websocket.send_json(result)
Resource efficiency
Approach	Containers	Memory	Startup	Max parallel
HTTP (1 env = 1 container)	N	N × ~100MB	N × ~5s	Limited by containers
WebSocket (N sessions = 1 container)	1	~200MB	~5s	Limited by MAX_CONCURRENT_ENVS
Configure session limits via environment variable:

docker run -d -p 8000:8000 -e MAX_CONCURRENT_ENVS=100 registry.hf.space/openenv-echo-env:latest
Scaling a Single Container
Before adding more containers, maximize the capacity of a single deployment. The key parameters are workers (CPU parallelism) and MAX_CONCURRENT_ENVS (session limit).

Uvicorn workers
Each Uvicorn worker is a separate process that can handle requests independently. More workers = more CPU cores utilized.

# Clone and run locally
git clone https://huggingface.co/spaces/burtenshaw/openenv-benchmark
cd openenv-benchmark
pip install -e .

# Run with 8 workers
WORKERS=8 uvicorn benchmark.server.app:app --host 0.0.0.0 --port 8000 --workers 8
The above example will use 8 workers and each worker will be able to handle 100 concurrent sessions. For simple environments, like text games, it's possible to get to 2000 concurrent sessions with 8 workers.

Note: More workers consume more memory. Each worker loads a full copy of the environment code.

Docker with environment variables
Pass scaling parameters when starting the container:

# Pull from HF Spaces registry
docker pull registry.hf.space/burtenshaw-openenv-benchmark:latest

# Run with custom configuration
docker run -d -p 8000:8000 \
    -e WORKERS=8 \
    -e MAX_CONCURRENT_ENVS=400 \
    --name openenv-benchmark \
    registry.hf.space/burtenshaw-openenv-benchmark:latest
Variable	Default	Description
WORKERS	4	Uvicorn worker processes
MAX_CONCURRENT_ENVS	100	Max WebSocket sessions per worker
PORT	8000	Server port
HOST	0.0.0.0	Bind address
HF Spaces configuration
Now, let's deploy the environment to HF Spaces so that we can interact with the server from the client. Configure scaling via Space Settings > Variables:

Go to your Space settings page
Add environment variables:
WORKERS=4 (max 4 on free tier, 8 on CPU Upgrade)
MAX_CONCURRENT_ENVS=100
Restart the Space
Tier	vCPU	Recommended workers	Expected max batch (textarena)
CPU Basic (Free)	2	2	~128
CPU Upgrade	8	4-8	~512
Limitation: HF Spaces free users tier caps at ~128 concurrent sessions regardless of configuration. See Scaling Experiments for measured limits.

Scaling limits
The experiments below found that even on larger instances, a single container eventually fails to scale and we need multiple containers to handle the load. For example, on a CPU Upgrade instance with 8 workers, the max batch was 1024 concurrent sessions:

Success rate drops to 92%
P99 latency exceeds 2× the expected step time
Connection errors increase under load
When this happens, we need to scale to multiple containers and use a load balancer.

For high-throughput workloads, scale horizontally by running multiple environment containers behind a load balancer.

Scenario	Recommended approach
Development / testing	Single container with WebSocket sessions
Moderate load (< 100 concurrent)	Single container, increase MAX_CONCURRENT_ENVS
High load (100+ concurrent)	Multiple containers + load balancer
GPU environments	One container per GPU
We explored this in detail in the Scaling Experiments repository.

Envoy configuration
Scaling expectations
Scaling Expectations

Setup	Containers	Sessions/container	Total capacity	Throughput
Single	1	100	100	~100 req/s
4× containers	4	100	400	~350 req/s
8× containers	8	100	800	~600 req/s
Note: Actual throughput depends on environment complexity and hardware. Benchmark your specific workload.

Experiments Results
This section documents experiments measuring OpenEnv scaling characteristics across five infrastructure configurations. Full experiment data and code available at burtenshaw/openenv-scaling.

Experiment setup
Benchmark environment: A minimal OpenEnv environment with configurable wait time (simulates computation). Each step() call sleeps for the specified duration, isolating infrastructure overhead from environment logic.

Infrastructure tested:

Infrastructure	Cores	Configuration
local-uvicorn	8	Direct Uvicorn, 8 workers
local-docker	8	Docker container from HF Spaces image
hf-spaces	2	HF Spaces free tier (cpu-basic)
slurm-single	48	Single AWS HPC node
slurm-multi	96	Two AWS HPC nodes + Envoy load balancer
Protocol: WebSocket (/ws) and HTTP (/reset, /step) compared where available.

Metrics:

Max batch: Largest concurrent request count with ≥95% success rate
Batch/core: Max batch divided by available cores (efficiency metric)
P99 latency: 99th percentile total request time
RPS: Requests per second at max batch
Results summary
Infrastructure	Max Batch (WS)	Cores	Batch/Core	P99 Latency	RPS
slurm-multi	16,384	96	170.7	29.8s	518
local-uvicorn	2,048	8	256.0	1.97s	932
local-docker	2,048	8	256.0	2.90s	682
slurm-single	512	48	10.7	1.45s	358
hf-spaces	128	2	64.0	2.68s	48
All results measured with wait=10.0s step duration.

Max Batch Comparison Maximum batch size by infrastructure (95% success threshold)

Finding 1: Local deployments have highest per-core efficiency
Single instance of Python and Docker both achieve 256 concurrent sessions per core—the highest efficiency observed. With 8 workers, both reach 2,048 concurrent sessions before degradation begins.

This makes sense because the environment is running in a single process and the overhead of the environment is relatively low. But it's ideal for hackers and developers who want to test their environment quickly or train on a single machine.

Batch Size	Success Rate	P99 Latency	Notes
32	100%	1.05s	Perfect scaling
128	100%	1.07s	Perfect scaling
512	100%	1.33s	Perfect scaling
2,048	96.5%	1.97s	Max reliable batch
4,096	63.8%	3.20s	Connection failures begin
8,192	36.9%	5.75s	Above capacity
Beyond 2,048 concurrent connections, success rate drops sharply. The failure mode is connection rejection, not timeout—the server saturates its connection pool.

Batch Per Core Per-core efficiency comparison across infrastructures

Finding 2: HF Spaces works reliably up to 128 concurrent sessions
HF Spaces free tier (cpu-basic) provides 2 workers and achieves 128 concurrent WebSocket sessions with 100% success. This translates to 64 sessions per core.

HF Spaces scaling behavior (WebSocket):

Batch Size	Success Rate	P99 Latency	Notes
1	100%	1.64s	Baseline
32	100%	1.80s	Perfect scaling
64	100%	2.14s	Perfect scaling
128	100%	2.68s	Max reliable batch
256	~33%	4.41s	Inconsistent (some runs 0%, some 100%)
512	0%	—	Complete failure
At 256 concurrent connections, results become unstable. At 512+, connections fail entirely due to HF Spaces connection limits.

HTTP mode does not work on HF Spaces. The /reset and /step HTTP endpoints are not accessible on the deployed Space—all HTTP requests fail. Use WebSocket mode exclusively.

Finding 3: Multi-node scaling works
Multi-node SLURM (96 cores across 2 nodes) achieves 16,384 concurrent sessions with 100% success rate—the highest absolute throughput tested.

SLURM multi-node scaling behavior:

Batch Size	Success Rate	P99 Latency	Notes
32	100%	1.05s	Perfect scaling
512	100%	1.59s	Perfect scaling
2,048	100%	3.48s	Perfect scaling
4,096	100%	6.97s	Perfect scaling
8,192	100%	13.7s	Perfect scaling
16,384	100%	29.8s	Max tested batch
The batch/core ratio (170.7) is lower than local deployments (256) but provides the highest absolute capacity for large-scale workloads.

Scaling Comparison

Multi-node vs single-node scaling behavior

Latency breakdown
At max load (wait=1.0s), latency breaks down as:

Infrastructure	Connect P50	Reset P50	Step P50	Total P99
slurm-single	0.26s	0.04s	1.00s	1.33s
local-uvicorn	0.58s	0.08s	1.05s	1.95s
hf-spaces	0.79s	0.10s	1.10s	2.48s
local-docker	1.38s	0.19s	1.05s	2.90s
slurm-multi	17.5s	2.25s	2.42s	26.3s
Observations:

Step latency is consistent across infrastructures (~1.0s for 1.0s wait), confirming the benchmark measures infrastructure overhead accurately
Connect latency varies significantly—local Docker shows higher connect time at load (1.38s), likely due to container networking
Multi-node has high connect latency (17.5s) at 16,384 batch due to queuing at the load balancer; this is the cost of handling 16× more connections than single-node
Latency Heatmap P99 latency across configurations and batch sizes

Scaling Curves Success rate vs batch size for all infrastructures

Test methodology
# Clone benchmark environment
git clone https://huggingface.co/spaces/burtenshaw/openenv-scaling
cd openenv-scaling

# Run scaling test
python tests/test_scaling.py \
    --url http://localhost:8000 \
    --requests-grid 32,128,512,2048,4096,8192,16384 \
    --wait-grid 1.0,5.0,10.0 \
    --reps 3 \
    --mode ws \
    --output-dir experiments/results/
Each configuration was tested with 3 repetitions. Max batch is defined as the largest batch size achieving ≥95% success rate across all repetitions.

Summary
Infrastructure	Best for	Max concurrent	Batch/core
local-uvicorn	Development, <2K sessions	2,048	256
local-docker	Same as uvicorn, containerized	2,048	256
hf-spaces	Demos, moderate load	128	64
slurm-single	HPC, single-node jobs	512	10.7
slurm-multi	Large-scale training	16,384	170.7
Recommendations:

For development and moderate workloads (<2,000 concurrent): Use single node Uvicorn or Docker depending software environment. These provide the best per-core efficiency (256 sessions/core).

For demos, testing, and published environments: HF Spaces free tier works reliably up to 128 concurrent sessions.

For large-scale training (>2,000 concurrent): Deploy multi-node with proper load balancing. Expect ~170 sessions per core, but much higher absolute throughput.

penEnv Wordle with GRPO using TRL
Open In Colab

trl banner

With Transformers Reinforcement Learning (TRL), you can train a model that learns to play Wordle, a word-guessing game, through interaction and reinforcement.

TRL GitHub Repository
Official TRL Examples
Community Tutorials
OpenEnv
An agentic environment is a setting where a model can take actions, observe outcomes, and adjust its behavior based on feedback, similar to how humans learn from trial and error. In this case, the agent interacts with the Wordle environment through the OpenEnv framework, which standardizes multi-agent and RL-style text environments.

Wordle is a popular word puzzle where the player must guess a secret five-letter word within six tries. After each guess, feedback indicates whether each letter is:

🟩 Correct and in the right position
🟨 Present but in the wrong position
⬛ Not in the word
This feedback loop makes Wordle a perfect environment for RL with LLMs, where the goal is to maximize the probability of guessing the correct word efficiently.

We will fine-tune a model using GRPO (Group Relative Policy Optimization) via TRL. The agent will:

Generate guesses based on the game state and feedback.
Receive structured feedback from the environment after each guess.
Learn to improve its guessing strategy over time through reward signals.
Install dependencies
We will start by installing TRL, which automatically includes the main dependencies like Transformers. We will also install the OpenEnv framework (for the environment), trackio (for logging and monitoring training runs), and vLLM (for efficient generation).

!pip install -Uq git+https://github.com/huggingface/trl.git git+https://github.com/meta-pytorch/OpenEnv.git trackio vllm==0.10.2 bitsandbytes
Log in to Hugging Face
Log in to your Hugging Face account to save your fine-tuned model, track your experiment results directly on the Hub or access gated models. You can find your access token on your account settings page.

from huggingface_hub import notebook_login

notebook_login()
Initialize the Environment
Let us begin by setting up the environment that will be used during training. For this task, we will rely on the TextArena environment from OpenEnv, which exposes a familiar Gymnasium-style API (reset(), step(), etc.) to simplify interaction.

In this example, we will connect to the hosted environment at burtenshaw/textarena. For production use or custom configurations, we strongly recommend running the environment locally via Docker. The hosted versions on the Hub currently have limited concurrency support, so duplicating the Space to your own account is the preferred approach in those cases.

For more information, refer to the TRL-OpenEnv documentation.

from envs.textarena_env import TextArenaEnv

textarena_url = "https://burtenshaw-textarena.hf.space" # Duplicate the Space and update this!
env = TextArenaEnv(base_url=textarena_url)
Init model and tokenizer
We will use Qwen/Qwen3-1.7B, a lightweight instruction-tuned model that works well for quick experiments. Despite its small size, it can still learn interesting strategies during fine-tuning. If you have stronger hardware, you can easily scale up to larger models.

from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
Rollout function with helpers
The rollout function defines how the agent interacts with the environment during GRPO training. It is responsible for generating model completions, collecting feedback (rewards), and returning all necessary information for optimization.

In this setup:

The function is called automatically by the GRPOTrainer during each training step.
It uses the trainer's built-in generate_rollout_completions() method for efficient generation with vLLM in colocate mode.
Each rollout represents a full interaction loop. The model guesses, receives feedback from Wordle, and updates based on reward signals.
System Prompt
First, we define the system_prompt that guides the model's behavior as an expert Wordle solver with strategic reasoning and structured responses.

system_prompt = """
You are an expert Wordle solver with deep knowledge of English vocabulary, letter frequency patterns, and optimal guessing strategies.

## GAME RULES

1. The target is a 5-letter English word
2. You have 6 attempts to guess the correct word
3. After each guess, you receive color-coded feedback:
   - GREEN: Letter is correct and in the correct position
   - YELLOW: Letter is in the word but in the wrong position
   - GRAY: Letter is not in the word at all
4. All guesses must be valid 5-letter English words
5. You cannot reuse a word you've already guessed

## RESPONSE FORMAT

Only respond with your next guess in square brackets, e.g., [crane].

## STRATEGIC APPROACH

Do not repeat the same guess twice.

### Opening Strategy
- Start with words rich in common vowels (A, E, I, O, U) and consonants (R, S, T, L, N)
- Optimal starters: CRANE, SLATE, STARE, AROSE, IRATE

### Mid-Game Strategy
- Use confirmed GREEN letters in their correct positions
- Place YELLOW letters in different positions than where they appeared
- Eliminate GRAY letters from consideration

## YOUR GOAL

Solve the Wordle in as few guesses as possible by strategically using feedback to eliminate impossible words and narrow down the solution space efficiently.
"""
Rollout Function
def rollout_func(prompts, trainer=None):
    """
    Rollout function for GRPO training with environment interaction.
    """
    episode_prompt_ids = []
    episode_completion_ids = []
    episode_logprobs = []
    correctness_rewards = []
    green_rewards = []
    yellow_rewards = []
    repetition_rewards = []

    for prompt_text in prompts:
        episode = rollout_once(
            trainer=trainer,
            env=env,
            tokenizer=tokenizer,
            dataset_prompt=prompt_text,
            system_prompt=system_prompt,
            max_turns=6,
        )
        episode_prompt_ids.append(episode["prompt_ids"])
        episode_completion_ids.append(episode["completion_ids"])
        episode_logprobs.append(episode["logprobs"])
        correctness_rewards.append(episode["correct_reward"])
        green_rewards.append(episode["green_reward"])
        yellow_rewards.append(episode["yellow_reward"])
        repetition_rewards.append(episode["repetition_reward"])

    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "correct_reward": correctness_rewards,
        "green_reward": green_rewards,
        "yellow_reward": yellow_rewards,
        "repetition_reward": repetition_rewards,
    }
Define rollout_once
The rollout_once function runs one full interaction loop between the model and the Wordle environment using the trainer's generation method.

from collections import defaultdict
from envs.textarena_env import TextArenaAction
from envs.textarena_env.rewards import extract_feedback_counts, extract_guess, extract_wordle_feedback
from trl.experimental.openenv import generate_rollout_completions


def rollout_once(trainer, env, tokenizer, dataset_prompt, system_prompt, max_turns):
    """
    Execute one full Wordle episode with the model.
    """
    result = env.reset()
    observation = result.observation

    prompt_ids = []
    completion_ids = []
    logprobs = []
    raw_rewards = []
    green_scores = []
    yellow_scores = []
    repetition_scores = []
    correct_scores = []
    guess_counts = defaultdict(int)

    for _turn in range(max_turns):
        if result.done:
            break

        base_prompt = observation.prompt or dataset_prompt
        user_prompt = make_user_prompt(base_prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])
        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        guess = extract_guess(completion_text)
        result = env.step(TextArenaAction(message=guess))
        raw_rewards.append(float(result.reward or 0.0))
        observation = result.observation
        correct_score = float(result.reward or 0.0)
        feedback = extract_wordle_feedback(observation)

        previous_occurrences = guess_counts[guess]
        repetition_score = scale_repetition_score(previous_occurrences, len(guess_counts))
        guess_counts[guess] += 1

        if not feedback:
            green_score = 0.0
            yellow_score = 0.0
        else:
            green_count, yellow_count = extract_feedback_counts(feedback)
            green_score = green_count / 5.0
            yellow_score = yellow_count / 5.0

        repetition_scores.append(repetition_score)
        green_scores.append(green_score)
        yellow_scores.append(yellow_score)
        correct_scores.append(correct_score)

    correct_reward_value = correct_scores[-1] if correct_scores else (raw_rewards[-1] if raw_rewards else 0.0)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "raw_rewards": raw_rewards,
        "correct_reward": correct_reward_value,
        "green_reward": green_scores[-1] if green_scores else 0.0,
        "yellow_reward": yellow_scores[-1] if yellow_scores else 0.0,
        "repetition_reward": repetition_scores[-1] if repetition_scores else 0.0,
    }
Helper functions
def make_user_prompt(prompt_text, messages):
    """Builds a structured user prompt combining the task description and message history"""
    history = format_history(messages)
    prompt_section = prompt_text.strip() if prompt_text.strip() else "Wordle-v0"
    history_section = history if history else "[PROMPT] Awaiting first feedback."
    return (
        f"Game prompt:\n{prompt_section}\n\n"
        f"Conversation so far:\n{history_section}\n\n"
        "Reply with your next guess enclosed in square brackets."
    )

def format_history(messages):
    """Formats the message history with tags for clear conversational context"""
    lines = []
    for message in messages:
        tag = message.category or "MESSAGE"
        content = message.content.strip()
        if not content:
            continue
        lines.append(f"[{tag}] {content}")
    return "\n".join(lines)

def scale_repetition_score(previous_occurrences, max_occurrences):
    """Scale the repetition score based on the number of previous occurrences from 0 to 1"""
    if max_occurrences == 0:
        return 0.0
    return (max_occurrences - previous_occurrences) / max_occurrences
Define reward functions
def reward_correct(completions, **kwargs):
    rewards = kwargs.get("correct_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_greens(completions, **kwargs):
    rewards = kwargs.get("green_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_yellows(completions, **kwargs):
    rewards = kwargs.get("yellow_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]


def reward_repetition(completions, **kwargs):
    rewards = kwargs.get("repetition_reward") if kwargs else None
    if rewards is None:
        return [0.0 for _ in completions]
    return [float(r) for r in rewards]
Create dataset
from datasets import Dataset

dataset_size = 1000
dataset_prompt = "Play Wordle like an expert."

dataset = Dataset.from_dict({"prompt": [dataset_prompt] * dataset_size})
Set GRPO Config
from trl import GRPOConfig

output_dir = "wordle-grpo-Qwen3-1.7B"

grpo_config = GRPOConfig(
    num_train_epochs = 1,
    learning_rate = 5e-6,
    gradient_accumulation_steps = 64,
    per_device_train_batch_size = 1,
    warmup_steps = 20,
    num_generations = 2,
    max_completion_length = 8,
    max_prompt_length = 1400,
    use_vllm = True,
    vllm_mode = "colocate",
    vllm_gpu_memory_utilization = 0.1,
    output_dir = output_dir,
    report_to="trackio",
    trackio_space_id = output_dir,
    logging_steps = 1,
    save_steps = 10,
    gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    push_to_hub = True,
)
Create GRPOTrainer and start training
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model_name,
    processing_class=tokenizer,
    reward_funcs=[
        reward_correct,
        reward_greens,
        reward_yellows,
        reward_repetition,
    ],
    train_dataset=dataset,
    args=grpo_config,
    rollout_func=rollout_func,
)
Memory stats before training
import torch
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")
Output:

GPU = NVIDIA A100-SXM4-40GB. Max memory = 39.557 GB.
10.516 GB of memory reserved.
Train!
trainer_stats = trainer.train()
Training Progress:

Step	Training Loss
1	0.008300
2	0.001900
3	0.015100
4	0.008700
5	0.009800
6	0.006700
7	0.006100
8	0.004400
9	-0.002100
10	0.007500
11	0.008400
12	0.008000
13	0.007800
14	-0.002400
15	-0.003200
16	-0.006000
17	-0.008300
18	-0.011000
19	-0.004200
20	-0.001700
21	-0.004100
22	-0.011600
23	-0.006400
24	-0.009100
25	0.003200
26	0.005100
27	-0.002800
28	0.001400
29	0.011500
30	-0.010500
31	-0.006400
Memory stats after training
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_training = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
training_memory_percentage = round(used_memory_for_training / max_memory * 100, 3)

print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_training} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {training_memory_percentage} %.")
Output:

5231.7046 seconds used for training.
87.2 minutes used for training.
Peak reserved memory = 36.68 GB.
Peak reserved memory for training = 26.164 GB.
Peak reserved memory % of max memory = 92.727 %.
Peak reserved memory for training % of max memory = 66.143 %.
Save and push to Hub
env.close()
trainer.save_model(output_dir)
trainer.push_to_hub()
Load the Fine-Tuned Model and Run Inference
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "sergiopaniego/wordle-grpo-Qwen3-1.7B" # Replace with your HF username

fine_tuned_model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
MAX_TURNS=6

def play_wordle(env, model, tokenizer):
    result = env.reset()
    observation = result.observation

    print("Initial Prompt:\n" + observation.prompt)

    for turn in range(MAX_TURNS):
        if result.done:
            break

        user_prompt = make_user_prompt(observation.prompt, observation.messages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )

        model_inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]

        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        guess = extract_guess(generated_text)

        print(f"\nTurn {turn}: model replied with -> {generated_text}")
        print(f"   Parsed guess: {guess}")

        result = env.step(TextArenaAction(message=guess))
        observation = result.observation

        print("   Feedback messages:")
        for message in observation.messages:
            print(f"     [{message.category}] {message.content}")

    print("\nGame finished")
    print(f"   Reward: {result.reward}")
    print(f"   Done: {result.done}")
Let us play the game!
try:
    play_wordle(env, fine_tuned_model, tokenizer)
finally:
    env.close()
Output:

Initial Prompt:
You are Player 0 in Wordle.
A secret 5-letter word has been chosen. You have 6 attempts to guess it.
For each guess, wrap your word in square brackets (e.g., [apple]).
Feedback for each letter will be given as follows:
  - G (green): correct letter in the correct position
  - Y (yellow): letter exists in the word but in the wrong position
  - X (wrong): letter is not in the word
Enter your guess to begin.

Turn 0: model replied with -> [crane]
   Parsed guess: [crane]
   Feedback messages:
     [MESSAGE] [crane]
     [MESSAGE] Player 0 submitted [crane].
Feedback:
C R A N E
X Y X X X

You have 5 guesses left.

Turn 1: model replied with -> [spare]
   Parsed guess: [spare]
   Feedback messages:
     [MESSAGE] [spare]
     [MESSAGE] Player 0 submitted [spare].
Feedback:
C R A N E
X Y X X X

S P A R E
G X X G X

You have 4 guesses left.

...

Game finished
   Reward: 0.0
   Done: True
Note: The model has learned some good opening strategies (starting with "crane", then "spare"), but still tends to repeat guesses. This is a common challenge in RL training that can be improved with:

Longer training runs
Stronger repetition penalties
Better reward shaping
Larger models



