#!/usr/bin/env python3
"""
Greenhouse Climate Control — Complete Test Runner (CLI).

Runs both offline unit tests and HTTP API tests.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --offline    # Offline tests only (no server needed)
    python run_tests.py --api        # API tests only (server must be running)
    python run_tests.py --api-start  # Start server, run API tests, then stop

Requirements:
    pip install requests

Server command (if needed):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 2
"""

import sys
import os
import argparse
import subprocess
import time
import signal
from pathlib import Path

# Ensure we can import from the greenhouse package
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run_offline_tests() -> bool:
    """Run environment-level unit tests (no server needed)."""
    from tests.test_environment import run_all as run_offline
    return run_offline()


def check_server(url: str = "http://127.0.0.1:8000") -> bool:
    """Check if server is reachable."""
    try:
        import requests
        r = requests.get(f"{url}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def start_server(port: int = 8000) -> subprocess.Popen:
    """Start the uvicorn server as a background process."""
    print(f"\n  🚀 Starting server on port {port}...")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "server.app:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--workers", "1",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to start
    for i in range(15):
        time.sleep(1)
        if check_server(f"http://127.0.0.1:{port}"):
            print(f"  ✅ Server started (PID={proc.pid})")
            return proc
        if proc.poll() is not None:
            # Server process died
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            print(f"  ❌ Server failed to start: {stderr[:300]}")
            return None
    print("  ❌ Server did not become ready in time")
    proc.terminate()
    return None


def stop_server(proc: subprocess.Popen):
    """Stop the server process."""
    if proc and proc.poll() is None:
        print(f"\n  🛑 Stopping server (PID={proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  ✅ Server stopped")


def run_api_tests() -> bool:
    """Run HTTP API tests (server must be running)."""
    from tests.test_api import run_all as run_api
    return run_api()


def main():
    parser = argparse.ArgumentParser(
        description="Greenhouse Climate Control — Test Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Run only offline unit tests (no server needed)",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Run only API tests (server must be running)",
    )
    parser.add_argument(
        "--api-start", action="store_true",
        help="Start server automatically, run API tests, then stop",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Server port (default: 8000)",
    )
    args = parser.parse_args()

    # Default: run everything
    run_offline_flag = not args.api
    run_api_flag = not args.offline
    auto_server = args.api_start

    print()
    print("╔" + "═" * 68 + "╗")
    print("║  🌿 Greenhouse Climate Control — Test Suite                       ║")
    print("╚" + "═" * 68 + "╝")

    results = {}

    # ── Offline Tests ────────────────────────────────────────────────
    if run_offline_flag and not args.api:
        print("\n\n📦 PHASE 1: Offline Unit Tests\n")
        results["offline"] = run_offline_tests()

    # ── API Tests ────────────────────────────────────────────────────
    if run_api_flag and not args.offline:
        print("\n\n🌐 PHASE 2: HTTP API Tests\n")

        server_proc = None
        if auto_server:
            server_proc = start_server(args.port)
            if server_proc is None:
                results["api"] = False
                print("  ❌ Skipping API tests (server failed to start)")
            else:
                results["api"] = run_api_tests()
                stop_server(server_proc)
        else:
            if check_server(f"http://127.0.0.1:{args.port}"):
                results["api"] = run_api_tests()
            else:
                print("  ❌ Server is not running!")
                print()
                print("  To run API tests, either:")
                print(f"    1. Start the server first:")
                print(f"       uvicorn server.app:app --host 0.0.0.0 --port {args.port}")
                print(f"    2. Or use --api-start to auto-start:")
                print(f"       python run_tests.py --api-start")
                print()
                results["api"] = False

    # ── Final Summary ────────────────────────────────────────────────
    print()
    print("╔" + "═" * 68 + "╗")
    print("║  FINAL SUMMARY                                                    ║")
    print("╠" + "═" * 68 + "╣")
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"║  {name:20s} {status:>46s} ║")
        if not passed:
            all_passed = False

    if all_passed and results:
        print("╠" + "═" * 68 + "╣")
        print("║  🎉 ALL TEST PHASES PASSED!                                      ║")
    elif not all_passed:
        print("╠" + "═" * 68 + "╣")
        print("║  ⚠️  SOME TESTS FAILED — see details above                       ║")
    print("╚" + "═" * 68 + "╝")
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
