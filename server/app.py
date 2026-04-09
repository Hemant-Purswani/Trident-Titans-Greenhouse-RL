# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Greenhouse Climate Control Environment.

This module creates an HTTP server that exposes the GreenhouseEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""
import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import GreenhouseAction, GreenhouseObservation
    from .greenhouse_environment import GreenhouseEnvironment
except (ImportError, SystemError):
    from models import GreenhouseAction, GreenhouseObservation
    from server.greenhouse_environment import GreenhouseEnvironment


# Create the app with web interface and README integration
app = create_app(
    GreenhouseEnvironment,
    GreenhouseAction,
    GreenhouseObservation,
    env_name="greenhouse",
    max_concurrent_envs=4,  # Support multiple concurrent WebSocket sessions
)

@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirect root to /docs or /web if ENABLE_WEB_INTERFACE is on."""
    from fastapi.responses import RedirectResponse
    # If ENABLE_WEB_INTERFACE is true, create_app serves the UI at root (/)
    # But if it's false, we redirect to /docs.
    if os.getenv("ENABLE_WEB_INTERFACE", "false").lower() in ("true", "1", "yes"):
        return RedirectResponse(url="/docs")
    return RedirectResponse(url="/docs")

@app.get("/tasks")
async def list_tasks():
    """Explicit endpoint for task discovery by the validator."""
    return GreenhouseEnvironment.TASKS

@app.get("/info")
async def get_info():
    """Returns general information about the environment and available tasks."""
    return {
        "env_name": "greenhouse",
        "tasks": GreenhouseEnvironment.TASKS,
        "spec_version": "1",
        "description": "Greenhouse Climate Control Environment"
    }


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    Reads HOST, PORT, and WORKERS from environment variables.
    Defaults: host=0.0.0.0, port=8000, workers=1

    Usage:
        uv run --project . server
        python -m greenhouse.server.app
        uvicorn server.app:app --workers 4
    """
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    workers = int(os.environ.get("WORKERS", "1"))

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
