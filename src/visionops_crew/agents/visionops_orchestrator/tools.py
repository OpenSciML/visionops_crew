"""Custom tools for the VisionOps Crew orchestrator agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.antigravity import Agent as AntigravityAgent
from google.antigravity import LocalAgentConfig
from google.antigravity.hooks import hooks, policy
from google.antigravity.types import (
    AntigravityConnectionError,
    AntigravityExecutionError,
    AntigravityValidationError,
    BuiltinTools,
)

ANTIGRAVITY_EXPERT_INSTRUCTIONS = """
You are a senior ML/code engineer and multimodal Antigravity super agent.

Use the local repository workspace to inspect files, analyze code, propose code,
draft notebook content, produce images or diagrams, draft validation reports,
and explain implementation tradeoffs. Keep work scoped to the user's request.

Do not create, edit, overwrite, or delete files unless the user explicitly
confirms the exact file-writing action in the current conversation. If a task
would benefit from file changes, first describe the proposed path and change,
then ask for confirmation. Without confirmation, return suggestions, patches, or
instructions as text only.

When asked to generate an image, use the `generate_image` tool. After the image
is created, tell the user the image name and a one-line confirmation. Do not
describe the image unless the user asks.
"""


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    """Resolve a path from a stable base directory.

    Args:
        path: Absolute path, home-relative path, or path relative to `base_dir`.
        base_dir: Directory used to resolve relative paths.

    Returns:
        Absolute normalized path.
    """
    path = path.expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_root_folder(root_folder: str) -> Path:
    """Resolve the Antigravity root folder from a tool argument or environment.

    Args:
        root_folder: Absolute folder supplied by the tool caller. If empty,
            `ANTIGRAVITY_ROOT_DIR` or the current working directory is used.

    Returns:
        Absolute root folder for environment files, workspace storage, and
        Antigravity artifacts.

    Raises:
        ValueError: If `root_folder` is provided as a relative path.
    """
    if root_folder:
        root_path = Path(root_folder).expanduser()
        if not root_path.is_absolute():
            raise ValueError("root_folder must be an absolute path when provided.")
        return root_path.resolve()

    env_root = os.getenv("ANTIGRAVITY_ROOT_DIR")
    if env_root:
        return _resolve_path(Path(env_root), base_dir=Path.cwd())

    return Path.cwd().resolve()


def _path_from_env(name: str, default: Path, *, base_dir: Path) -> Path:
    """Resolve an env-configurable path from the Antigravity root folder.

    Args:
        name: Environment variable name that may override `default`.
        default: Path used when the environment variable is unset.
        base_dir: Directory used to resolve relative override values.

    Returns:
        Absolute normalized path.
    """
    raw_value = os.getenv(name)
    path = Path(raw_value).expanduser() if raw_value else default
    return _resolve_path(path, base_dir=base_dir)


def _configure_antigravity_paths(root_folder: str) -> dict[str, Path]:
    """Configure Antigravity paths for one tool call.

    Args:
        root_folder: Absolute caller-supplied root folder, or an empty string to
            use the environment/default root resolution.

    Returns:
        Mapping of storage roles to resolved paths used by the Antigravity run.

    Raises:
        ValueError: If the supplied root folder is relative.
    """
    root_path = _resolve_root_folder(root_folder)
    env_file = _path_from_env(
        "ANTIGRAVITY_ENV_FILE",
        root_path / ".env",
        base_dir=root_path,
    )
    load_dotenv(env_file)

    workspace_dir = _path_from_env(
        "ANTIGRAVITY_WORKSPACE_DIR",
        root_path / "antigravity-workspace",
        base_dir=root_path,
    )
    artifacts_dir = _path_from_env(
        "ANTIGRAVITY_ARTIFACTS_DIR",
        root_path / ".antigravity",
        base_dir=root_path,
    )

    return {
        "root": root_path,
        "env_file": env_file,
        "workspace": workspace_dir,
        "artifacts": artifacts_dir,
        "app_data": artifacts_dir / "app-data",
        "sessions": artifacts_dir / "sessions",
    }


def _ensure_antigravity_dirs(paths: dict[str, Path]) -> None:
    """Create Antigravity workspace and storage directories.

    Args:
        paths: Resolved Antigravity path mapping from
            `_configure_antigravity_paths`.
    """
    paths["workspace"].mkdir(parents=True, exist_ok=True)
    paths["app_data"].mkdir(parents=True, exist_ok=True)
    (paths["sessions"] / "expert").mkdir(parents=True, exist_ok=True)


class ArtifactTrackingHook(hooks.PostToolCallHook):
    """Collect Antigravity tool-call outputs during a single agent run.

    Attributes:
        tool_results: Serialized records for each completed Antigravity tool
            call, including the tool name, error state, and JSON-compatible
            result payload when available.
    """

    def __init__(self) -> None:
        """Initialize an empty tool-result buffer for one agent invocation."""
        self.tool_results: list[dict[str, Any]] = []

    async def run(self, context: hooks.HookContext, data: Any) -> None:
        """Record a completed Antigravity tool call.

        Args:
            context: Hook execution context supplied by the Antigravity SDK.
            data: Post-tool-call payload containing the tool name, error state,
                and result object.
        """
        result = data.result
        record: dict[str, Any] = {
            "tool": str(data.name),
            "error": data.error,
        }
        if hasattr(result, "model_dump"):
            record["result"] = result.model_dump(mode="json")
        else:
            record["result"] = result

        self.tool_results.append(record)


def _build_policies(*, allow_file_edits: bool) -> list[policy.Policy]:
    """Build non-interactive Antigravity policies.

    Args:
        allow_file_edits: Whether Antigravity may create and edit files.

    Returns:
        Policy list that allows image generation and non-write tools while
        denying command execution.
    """
    file_edit_policies = (
        [
            policy.allow(BuiltinTools.CREATE_FILE.value, name="allow-create-file"),
            policy.allow(BuiltinTools.EDIT_FILE.value, name="allow-edit-file"),
        ]
        if allow_file_edits
        else [
            policy.deny(BuiltinTools.CREATE_FILE.value, name="deny-create-file"),
            policy.deny(BuiltinTools.EDIT_FILE.value, name="deny-edit-file"),
        ]
    )
    return [
        *file_edit_policies,
        policy.deny(BuiltinTools.RUN_COMMAND.value, name="deny-run-command"),
        policy.allow(BuiltinTools.GENERATE_IMAGE.value, name="allow-generate-image"),
        policy.allow("*", name="allow-other-tools"),
    ]


def _build_antigravity_config(
    *,
    paths: dict[str, Path],
    allow_file_edits: bool,
    extra_hooks: list[hooks.Hook],
) -> LocalAgentConfig:
    """Create a local Antigravity config for a tool call.

    Args:
        paths: Resolved storage paths for the Antigravity run.
        allow_file_edits: Whether file create/edit tools should be allowed.
        extra_hooks: Additional SDK hooks to attach to the agent session.

    Returns:
        Local Antigravity agent configuration.
    """
    _ensure_antigravity_dirs(paths)
    return LocalAgentConfig(
        system_instructions=ANTIGRAVITY_EXPERT_INSTRUCTIONS,
        model=os.getenv("ANTIGRAVITY_MODEL", "gemini-3.5-flash"),
        workspaces=[str(paths["workspace"])],
        api_key=os.getenv("GEMINI_API_KEY"),
        save_dir=str(paths["sessions"] / "expert"),
        app_data_dir=str(paths["app_data"]),
        policies=_build_policies(allow_file_edits=allow_file_edits),
        hooks=extra_hooks,
    )


def _artifact_roots(
    *,
    paths: dict[str, Path],
    include_workspace: bool,
) -> list[Path]:
    """Return roots that Antigravity can write artifacts into.

    Args:
        paths: Resolved storage paths for the Antigravity run.
        include_workspace: Whether to include the writable workspace directory.

    Returns:
        Artifact roots to scan before and after agent execution.
    """
    roots = [paths["artifacts"]]
    if include_workspace:
        roots.append(paths["workspace"])
    return roots


def _snapshot_files(roots: list[Path]) -> dict[str, int]:
    """Snapshot file mtimes under roots.

    Args:
        roots: Directories to scan recursively.

    Returns:
        Mapping from absolute file path strings to modification timestamps in
        nanoseconds.
    """
    snapshot: dict[str, int] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                snapshot[str(path)] = path.stat().st_mtime_ns
    return snapshot


def _diff_snapshots(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, list[str]]:
    """Return created and updated files between two snapshots.

    Args:
        before: File snapshot captured before an Antigravity run.
        after: File snapshot captured after an Antigravity run.

    Returns:
        Created and updated absolute file paths grouped by change type.
    """
    return {
        "created": sorted(path for path in after if path not in before),
        "updated": sorted(
            path
            for path, mtime in after.items()
            if path in before and before[path] != mtime
        ),
    }


async def run_antigravity_agent(
    task: str,
    root_folder: str = "",
    allow_file_edits: bool = False,
) -> dict[str, Any]:
    """Run Antigravity for an explicitly requested task.

    Args:
        task: Code, image, analysis, notebook, or artifact-generation request.
        root_folder: Absolute base folder for `.env`, `.antigravity/`, and
            `antigravity-workspace/`. If empty, `ANTIGRAVITY_ROOT_DIR` or the
            ADK process working directory is used.
        allow_file_edits: Whether to allow Antigravity create/edit file tools.
            Use only after the user confirms the exact file-writing action.

    Returns:
        Antigravity text output, resolved storage paths, detected artifacts, and
        captured tool results.

    Raises:
        ValueError: If `task` is blank.
        RuntimeError: If Antigravity configuration, connection, or execution
            fails.
    """
    if not task.strip():
        raise ValueError("task is required.")

    try:
        paths = _configure_antigravity_paths(root_folder)
        artifact_hook = ArtifactTrackingHook()
        roots = _artifact_roots(paths=paths, include_workspace=allow_file_edits)
        before_files = _snapshot_files(roots)

        async with AntigravityAgent(
            _build_antigravity_config(
                paths=paths,
                allow_file_edits=allow_file_edits,
                extra_hooks=[artifact_hook],
            )
        ) as agent:
            response = await agent.chat(task.strip())
            result = await response.text()

        after_files = _snapshot_files(roots)
        artifact_changes = _diff_snapshots(before_files, after_files)
        return {
            "status": "success",
            "result": result,
            "root_folder": str(paths["root"]),
            "workspace": str(paths["workspace"]),
            "artifact_root": str(paths["artifacts"]),
            "artifact_roots": [str(root) for root in roots],
            "created_artifacts": artifact_changes["created"],
            "updated_artifacts": artifact_changes["updated"],
            "tool_results": artifact_hook.tool_results,
        }
    except AntigravityValidationError as exc:
        raise RuntimeError(
            "Antigravity configuration is invalid. Check GEMINI_API_KEY, "
            "ANTIGRAVITY_MODEL, root_folder, and workspace settings."
        ) from exc
    except AntigravityConnectionError as exc:
        raise RuntimeError(
            "Antigravity connection dropped. Retry the request or check the "
            "local network/session state."
        ) from exc
    except AntigravityExecutionError as exc:
        raise RuntimeError(f"Antigravity execution failed: {exc}") from exc
