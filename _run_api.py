"""Quick API test runner that saves output."""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

old_stdout = sys.stdout
buf = io.StringIO()
sys.stdout = buf

from tests.test_api import run_all
success = run_all()

sys.stdout = old_stdout
output = buf.getvalue()

with open("api_test_log.txt", "w", encoding="ascii", errors="replace") as f:
    f.write(output)

# Print key lines
for line in output.splitlines():
    if any(k in line for k in ["FAIL", "RESULT", "PASS", "ERROR", "Server"]):
        print(line[:120])

print(f"\nOverall: {'PASS' if success else 'FAIL'}")
