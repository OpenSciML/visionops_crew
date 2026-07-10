"""Run Antigravity as a local code and image-generation expert.

Examples:
    uv run python scripts/antigravity_code_expert.py

Edit DEFAULT_TASK below to try different coding or image-generation prompts.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv
from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.types import (
    AntigravityConnectionError,
    AntigravityExecutionError,
    AntigravityValidationError,
)
from google.antigravity.utils import interactive

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
ANTIGRAVITY_ARTIFACTS_DIR = REPO_ROOT / ".antigravity"
ANTIGRAVITY_APP_DATA_DIR = ANTIGRAVITY_ARTIFACTS_DIR / "app-data"
ANTIGRAVITY_SESSIONS_DIR = ANTIGRAVITY_ARTIFACTS_DIR / "sessions"
logger = logging.getLogger(__name__)
StatusChangeHook = Callable[[str, str], None]
DEFAULT_TASK = (
    "Generate an image of a futuristic city skyline at sunset with flying cars."
)

ANTIGRAVITY_EXPERT_INSTRUCTIONS = """
You are a senior ML/code engineer, that can generate images.

Use the local workspace to inspect files, write clean Python code, create or edit
scripts, and run lightweight validation commands when useful. Stay scoped to
repo-local coding, scripting, debugging, and implementation tasks.

When asked to generate an image, use the 'generate_image' tool. After the image
is created, tell the user the image name and a one-line confirmation. Do not
describe the image.
"""


def ensure_antigravity_artifact_dirs() -> None:
    """Create repo-local Antigravity storage directories."""
    ANTIGRAVITY_APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (ANTIGRAVITY_SESSIONS_DIR / "expert").mkdir(parents=True, exist_ok=True)


def build_antigravity_config() -> LocalAgentConfig:
    """Create a local Antigravity config for coding and image tasks.

    Returns:
        Local Antigravity agent configuration with repo-local storage paths,
        environment-provided API credentials, and interactive policies.
    """
    ensure_antigravity_artifact_dirs()
    return LocalAgentConfig(
        system_instructions=ANTIGRAVITY_EXPERT_INSTRUCTIONS,
        model=os.getenv("ANTIGRAVITY_MODEL", "gemini-3.5-flash"),
        workspaces=[str(REPO_ROOT)],
        api_key=os.getenv("GEMINI_API_KEY"),
        save_dir=str(ANTIGRAVITY_SESSIONS_DIR / "expert"),
        app_data_dir=str(ANTIGRAVITY_APP_DATA_DIR),
        policies=[
            policy.ask_user(
                "generate_image",
                handler=interactive.ask_user_handler,
                name="allow-gen",
            ),
            policy.confirm_run_command(handler=interactive.ask_user_handler),
        ],
        hooks=[interactive.AskQuestionHook()],
    )


def log_status_change(status: str, message: str) -> None:
    """Default status hook for direct script runs.

    Args:
        status: Short task lifecycle state such as `running` or `completed`.
        message: Human-readable status detail to write to the logger.
    """
    logger.info("%s: %s", status, message)


async def run_antigravity_task(
    config: LocalAgentConfig,
    task: str,
    on_status_change: StatusChangeHook | None = None,
) -> str:
    """Run one Antigravity turn and return the final text response.

    Args:
        config: Local Antigravity configuration for the agent session.
        task: User request to send to Antigravity.
        on_status_change: Optional callback invoked when the task starts,
            completes, fails, or is cancelled.

    Returns:
        Final response text, or an empty string when the task is cancelled.

    Raises:
        AntigravityValidationError: If the SDK configuration is invalid.
        ValueError: If configuration values fail local validation.
        AntigravityConnectionError: If the SDK connection drops.
        AntigravityExecutionError: If agent execution fails.
    """
    try:
        if on_status_change:
            on_status_change("running", "Starting Antigravity task.")

        async with Agent(config) as agent:
            response = await agent.chat(task)
            result = await response.text()

        if on_status_change:
            on_status_change("completed", "Antigravity task completed.")
        return result
    except AntigravityValidationError as exc:
        logger.error("Configuration error: %s", exc)
        if on_status_change:
            on_status_change("failed", f"Configuration error: {exc}")
        raise
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        if on_status_change:
            on_status_change("failed", f"Configuration error: {exc}")
        raise
    except AntigravityConnectionError as exc:
        logger.error("SDK connection dropped: %s", exc)
        if on_status_change:
            on_status_change("failed", f"Connection dropped: {exc}")
        raise
    except AntigravityExecutionError as exc:
        logger.error("Agent execution failed: %s", exc)
        if on_status_change:
            on_status_change("failed", f"Execution failed: {exc}")
        raise
    except asyncio.CancelledError:
        logger.warning("Audit cancelled via shutdown signal.")
        if on_status_change:
            on_status_change("cancelled", "Audit cancelled.")
        return ""


async def main() -> None:
    """Run the default Antigravity demo task and print the final response."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    config = build_antigravity_config()
    result = await run_antigravity_task(
        config,
        DEFAULT_TASK,
        on_status_change=log_status_change,
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
