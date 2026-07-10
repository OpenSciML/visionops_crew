import os
from pathlib import Path

from dotenv import load_dotenv
from google.adk.code_executors import BuiltInCodeExecutor, UnsafeLocalCodeExecutor
from google.adk.agents.llm_agent import Agent
from google.adk.apps.app import App
from google.adk.environment import LocalEnvironment
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.environment import EnvironmentToolset
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StdioServerParameters,
)
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset

from .prompts import (
    ML_CODE_EXECUTION_AGENT_INSTRUCTION,
    ML_ENGINEER_AGENT_INSTRUCTION,
)

workspace_root = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)
load_dotenv(workspace_root / ".env")

local_environment = LocalEnvironment(
    working_dir=workspace_root,
    env_vars=dict(os.environ),
)

jupyter_mcp_env = {
    **dict(os.environ),
    "JUPYTER_URL": os.getenv("JUPYTER_URL", "http://localhost:4040"),
    "JUPYTER_TOKEN": os.getenv("JUPYTER_TOKEN", ""),
    "ALLOW_IMG_OUTPUT": os.getenv("ALLOW_IMG_OUTPUT", "true"),
}

# MCP toolset
jupyter_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uvx",
            args=[
                "jupyter-mcp-server@latest",
            ],
            env=jupyter_mcp_env
        ),
        timeout=30.0
    )
)

base_root = Path(__file__).resolve().parent

# Load the Jupyter and Hardware skills
jupyter_skill = load_skill_from_dir(
    base_root / "skills" / "jupyter-notebook"
)
hardware_skill = load_skill_from_dir(
    base_root / "skills" / "hardware-profiling"
)

# Skill toolset
skill_toolset = SkillToolset(
    skills=[jupyter_skill, hardware_skill],
    additional_tools=[
        jupyter_mcp_toolset,
    ],
    code_executor=UnsafeLocalCodeExecutor(),
)

code_execution_agent = Agent(
    model=os.getenv("ML_CODE_EXECUTION_MODEL", "gemini-3.5-flash"),
    name="ml_code_execution",
    description=(
        "Executes short Python snippets for calculations, data checks, "
        "algorithm experiments, and small ML/debugging probes."
    ),
    instruction=ML_CODE_EXECUTION_AGENT_INSTRUCTION,
    code_executor=BuiltInCodeExecutor(),
)

root_agent = Agent(
    model=os.getenv("ML_ENGINEER_MODEL", "gemini-3.5-flash"),
    name="ml_engineer",
    description=(
        "A machine learning engineer that can design, implement, debug, and "
        "run model training workflows using PyTorch, scikit-learn, JAX, and "
        "related Python ML tooling."
    ),
    instruction=ML_ENGINEER_AGENT_INSTRUCTION,
    tools=[
        skill_toolset,
        AgentTool(agent=code_execution_agent),
        EnvironmentToolset(
            environment=local_environment,
            max_output_chars=20000,
        ),
    ],
)

app = App(
    name="ml_engineer",
    root_agent=root_agent,
    plugins=[
        ReflectAndRetryToolPlugin(max_retries=3),
    ],
)
