"""Draft A2A service that exposes Antigravity directly.

This is future-work scaffolding only. It intentionally lives under scripts and
is not wired into the package or docs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events.event_queue import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotificationConfigStore
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities
from a2a.types import AgentCard
from a2a.types import AgentProvider
from a2a.types import AgentSkill
from a2a.types import Artifact
from a2a.types import Message
from a2a.types import Role
from a2a.types import TaskArtifactUpdateEvent
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.types import TaskStatusUpdateEvent
from a2a.types import TextPart
from dotenv import load_dotenv
from google.antigravity import Agent as AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity.hooks import policy
from google.antigravity.types import (
    AntigravityConnectionError,
    AntigravityExecutionError,
    AntigravityValidationError,
    BuiltinTools,
)
from starlette.applications import Starlette

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8004
AGENT_NAME = "antigravity_a2a_service"
AGENT_VERSION = "0.1.0-draft"

ANTIGRAVITY_PROMPT = """
Act as a senior computer vision and ML engineering collaborator.

Analyze code, design implementation plans, draft patches, produce concise
technical reports, and generate images only when explicitly requested. Stay
scoped to the configured workspace and the user's request.

File creation and editing are disabled by default unless
ANTIGRAVITY_A2A_ALLOW_FILE_EDITS=true is set. If file edits are disabled,
return suggested changes or patches as text.
"""


def _env_bool(name: str, *, default: bool = False) -> bool:
    """Return a boolean from a permissive environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_path(value: str | None, *, default: Path, base_dir: Path) -> Path:
    """Resolve absolute, home-relative, or base-relative paths."""
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _timestamp() -> str:
    """Return an A2A timestamp string."""
    return datetime.now(timezone.utc).isoformat()


class AntigravityA2AWrapper(AgentExecutor):
    """Standalone A2A wrapper over the Antigravity SDK."""

    def __init__(self) -> None:
        """Initialize the wrapper, public A2A endpoint, and SDK paths."""
        self.host = os.getenv("ANTIGRAVITY_A2A_PUBLIC_HOST", DEFAULT_HOST)
        self.port = int(os.getenv("ANTIGRAVITY_A2A_PUBLIC_PORT", str(DEFAULT_PORT)))
        self.protocol = os.getenv("ANTIGRAVITY_A2A_PUBLIC_PROTOCOL", "http")
        root = _resolve_path(
            os.getenv("ANTIGRAVITY_A2A_ROOT_DIR"),
            default=SCRIPT_DIR,
            base_dir=SCRIPT_DIR,
        )
        env_file = _resolve_path(
            os.getenv("ANTIGRAVITY_A2A_ENV_FILE"),
            default=root / ".env",
            base_dir=root,
        )
        load_dotenv(env_file)

        workspace = _resolve_path(
            os.getenv("ANTIGRAVITY_A2A_WORKSPACE_DIR"),
            default=root / "antigravity-a2a-workspace",
            base_dir=root,
        )
        artifacts = _resolve_path(
            os.getenv("ANTIGRAVITY_A2A_ARTIFACTS_DIR"),
            default=root / ".antigravity-a2a",
            base_dir=root,
        )
        self.paths = {
            "root": root,
            "env_file": env_file,
            "workspace": workspace,
            "artifacts": artifacts,
            "app_data": artifacts / "app-data",
            "sessions": artifacts / "sessions",
        }
        self.allow_file_edits = _env_bool("ANTIGRAVITY_A2A_ALLOW_FILE_EDITS")

    @property
    def url(self) -> str:
        """Return the public A2A endpoint URL advertised in the agent card."""
        return f"{self.protocol}://{self.host}:{self.port}/"

    def _ensure_antigravity_dirs(self) -> None:
        """Create Antigravity folders used by this wrapper."""
        self.paths["workspace"].mkdir(parents=True, exist_ok=True)
        self.paths["app_data"].mkdir(parents=True, exist_ok=True)
        (self.paths["sessions"] / "a2a").mkdir(parents=True, exist_ok=True)

    def _build_policies(self) -> list[policy.Policy]:
        """Build non-interactive Antigravity policies for this wrapper."""
        write_policies = (
            [
                policy.allow(BuiltinTools.CREATE_FILE.value, name="allow-create-file"),
                policy.allow(BuiltinTools.EDIT_FILE.value, name="allow-edit-file"),
            ]
            if self.allow_file_edits
            else [
                policy.deny(BuiltinTools.CREATE_FILE.value, name="deny-create-file"),
                policy.deny(BuiltinTools.EDIT_FILE.value, name="deny-edit-file"),
            ]
        )
        return [
            *write_policies,
            policy.deny(BuiltinTools.RUN_COMMAND.value, name="deny-run-command"),
            policy.allow(BuiltinTools.GENERATE_IMAGE.value, name="allow-generate-image"),
            policy.allow("*", name="allow-other-tools"),
        ]

    def build_antigravity_config(self) -> LocalAgentConfig:
        """Create the Antigravity SDK config used by this wrapper."""
        self._ensure_antigravity_dirs()
        return LocalAgentConfig(
            system_instructions=ANTIGRAVITY_PROMPT,
            model=os.getenv("ANTIGRAVITY_MODEL", "gemini-3.5-flash"),
            workspaces=[str(self.paths["workspace"])],
            api_key=os.getenv("GEMINI_API_KEY"),
            save_dir=str(self.paths["sessions"] / "a2a"),
            app_data_dir=str(self.paths["app_data"]),
            policies=self._build_policies(),
        )

    def build_agent_card(self) -> AgentCard:
        """Build the A2A agent card for this wrapper."""
        return AgentCard(
            name=AGENT_NAME,
            description="Draft standalone A2A wrapper over the Antigravity SDK.",
            url=self.url,
            version=AGENT_VERSION,
            capabilities=AgentCapabilities(),
            skills=[
                AgentSkill(
                    id="antigravity_prompt",
                    name="Antigravity prompt execution",
                    description="Runs a user prompt through the configured Antigravity SDK agent.",
                    examples=[
                        "Analyze this training script and propose fixes.",
                        "Draft a patch for a computer vision dataset loader.",
                    ],
                    input_modes=["text/plain"],
                    output_modes=["text/plain"],
                    tags=["antigravity", "computer-vision", "ml-engineering"],
                )
            ],
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            supports_authenticated_extended_card=False,
            provider=AgentProvider(
                organization="VisionOps Crew",
                url="https://github.com/haruiz/visionops_crew",
            ),
        )

    def to_app(self) -> Starlette:
        """Create a Starlette app exposing this wrapper over A2A."""
        request_handler = DefaultRequestHandler(
            agent_executor=self,
            task_store=InMemoryTaskStore(),
            push_config_store=InMemoryPushNotificationConfigStore(),
        )
        a2a_app = A2AStarletteApplication(
            agent_card=self.build_agent_card(),
            http_handler=request_handler,
        )
        app = Starlette()
        a2a_app.add_routes_to_app(app)
        return app

    @staticmethod
    def _part_to_text(part: Any) -> str:
        """Best-effort text extraction for an A2A message part."""
        value = getattr(part, "root", part)
        text = getattr(value, "text", None)
        if text:
            return str(text)

        data = getattr(value, "data", None)
        if data is not None:
            return json.dumps(data, ensure_ascii=True, sort_keys=True)

        file_value = getattr(value, "file", None)
        if file_value is not None:
            uri = getattr(file_value, "uri", None)
            name = getattr(file_value, "name", None)
            return f"[file: {uri or name or 'attached file'}]"

        if hasattr(value, "model_dump_json"):
            return value.model_dump_json(exclude_none=True)
        return str(value)

    def message_to_prompt(self, message: Message) -> str:
        """Convert an A2A message into the prompt sent to Antigravity."""
        prompt_parts = [self._part_to_text(part) for part in message.parts]
        prompt = "\n\n".join(part for part in prompt_parts if part.strip()).strip()
        if not prompt:
            raise ValueError("A2A request must contain at least one text part.")
        return prompt

    async def run_antigravity_prompt(self, prompt: str) -> dict[str, Any]:
        """Run one Antigravity turn for an A2A prompt."""
        if not prompt.strip():
            raise ValueError("prompt is required.")

        try:
            async with AntigravityAgent(self.build_antigravity_config()) as agent:
                response = await agent.chat(prompt.strip())
                result = await response.text()
        except AntigravityValidationError as exc:
            raise RuntimeError(
                "Antigravity configuration is invalid. Check GEMINI_API_KEY, "
                "ANTIGRAVITY_MODEL, and ANTIGRAVITY_A2A_* path settings."
            ) from exc
        except AntigravityConnectionError as exc:
            raise RuntimeError("Antigravity connection dropped.") from exc
        except AntigravityExecutionError as exc:
            raise RuntimeError(f"Antigravity execution failed: {exc}") from exc

        return {
            "result": result,
            "workspace": str(self.paths["workspace"]),
            "artifact_root": str(self.paths["artifacts"]),
            "file_edits_enabled": self.allow_file_edits,
        }

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Report that cancellation is unsupported for this draft."""
        raise NotImplementedError("Cancellation is not supported by this draft.")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute an A2A request by running the Antigravity prompt."""
        if not context.message:
            raise ValueError("A2A request must have a message.")

        await self.enqueue_status(
            event_queue,
            context,
            state=TaskState.submitted,
            message=context.message,
            final=False,
        )
        await self.enqueue_status(
            event_queue,
            context,
            state=TaskState.working,
            final=False,
        )

        try:
            prompt = self.message_to_prompt(context.message)
            response = await self.run_antigravity_prompt(prompt)
        except Exception as exc:
            await self.enqueue_status(
                event_queue,
                context,
                state=TaskState.failed,
                text=str(exc),
                final=True,
            )
            return

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                last_chunk=True,
                artifact=Artifact(
                    artifact_id=str(uuid4()),
                    parts=[TextPart(text=response["result"])],
                    metadata={
                        "workspace": response["workspace"],
                        "artifact_root": response["artifact_root"],
                        "file_edits_enabled": response["file_edits_enabled"],
                    },
                ),
            )
        )
        await self.enqueue_status(
            event_queue,
            context,
            state=TaskState.completed,
            final=True,
        )

    async def enqueue_status(
        self,
        event_queue: EventQueue,
        context: RequestContext,
        *,
        state: TaskState,
        final: bool,
        message: Message | None = None,
        text: str | None = None,
    ) -> None:
        """Publish a TaskStatusUpdateEvent."""
        if text is not None:
            message = Message(
                message_id=str(uuid4()),
                role=Role.agent,
                parts=[TextPart(text=text)],
            )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                final=final,
                status=TaskStatus(
                    state=state,
                    timestamp=_timestamp(),
                    message=message,
                ),
            )
        )


def create_app() -> Starlette:
    """Create a Starlette app with direct A2A routes registered."""
    return AntigravityA2AWrapper().to_app()


def parse_args() -> argparse.Namespace:
    """Parse local development server arguments."""
    parser = argparse.ArgumentParser(
        description="Run the draft direct Antigravity A2A service."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("ANTIGRAVITY_A2A_HOST", DEFAULT_HOST),
        help="Bind host for the local service.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ANTIGRAVITY_A2A_PORT", str(DEFAULT_PORT))),
        help="Bind port for the local service.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload during local development.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the draft A2A service with uvicorn."""
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    args = parse_args()

    import uvicorn

    uvicorn.run(
        "scripts.antigravity_a2a_service:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
