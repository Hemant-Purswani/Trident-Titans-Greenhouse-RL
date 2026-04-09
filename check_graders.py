import sys
import os
from pathlib import Path

# Add root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from server.greenhouse_environment import GreenhouseEnvironment
    from models import GreenhouseAction
except ImportError:
    # If running from root without package structure
    from server.greenhouse_environment import GreenhouseEnvironment
    from models import GreenhouseAction

def test_task(task_id):
    print(f"\nTesting task: {task_id}")
    env = GreenhouseEnvironment(task_id=task_id)
    env.reset()
    
    # Run for a few steps
    for _ in range(5):
        action = GreenhouseAction()
        obs = env.step(action)
    
    # Trigger end of episode by manually setting step count if needed, 
    # but here we'll just run until max_steps
    max_steps = env._config["max_steps"]
    for _ in range(max_steps - 5):
        obs = env.step(GreenhouseAction())
    
    score = obs.metadata.get("grader_score")
    print(f"Final grader score: {score}")
    
    if score is None:
        print("❌ FAIL: No grader_score in metadata")
        return False
        
    if 0.0 < score < 1.0:
        print(f"✅ PASS: Score {score} is strictly between 0 and 1")
        return True
    else:
        print(f"❌ FAIL: Score {score} is NOT strictly between 0 and 1")
        return False

if __name__ == "__main__":
    tasks = GreenhouseEnvironment.TASKS
    print(f"Discovered tasks: {tasks}")
    all_passed = True
    for t in tasks:
        if not test_task(t):
            all_passed = False
            
    if all_passed:
        print("\n✨ ALL TASKS PASSED GRADIENT CHECK!")
        sys.exit(0)
    else:
        print("\n❌ SOME TASKS FAILED.")
        sys.exit(1)
