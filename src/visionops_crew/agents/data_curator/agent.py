import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools.mcp_tool.mcp_session_manager import (
    StdioConnectionParams,
    StdioServerParameters,
)
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset

from .prompts import (
    DATA_CURATOR_AGENT_INSTRUCTION,
)
from .tools import (
    load_huggingface_dataset,
    open_fiftyone_dataset,
)


# Load local settings such as DATA_CURATOR_MODEL and FiftyOne/MongoDB env vars.
load_dotenv(".env")

workspace_root = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").exists()
)

# Run the MCP server from the same Python environment as ADK.
fiftyone_mcp_path = Path(sys.executable).with_name("fiftyone-mcp")

# Expose the FiftyOne MCP server's tools to this ADK agent over stdio.
fiftyone_mcp_tools = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=str(fiftyone_mcp_path),
            env=dict(os.environ),
            cwd=str(workspace_root),
        ),
        timeout=30.0,
    ),
)

# ADK discovers this object as the runnable agent for the data_curator package.
root_agent = Agent(
    model=os.getenv(
        "DATA_CURATOR_MODEL",
        "gemini-3.5-flash"
    ),
    name="data_curator",
    description=(
        "A computer vision data curator that can import datasets, inspect "
        "schemas and labels, run dataset curation workflows, and work with a "
        "local FiftyOne App through direct import/open tools and FiftyOne MCP."
    ),
    instruction=DATA_CURATOR_AGENT_INSTRUCTION,
    tools=[
        load_huggingface_dataset,
        open_fiftyone_dataset,
        fiftyone_mcp_tools,
    ],
)

app = App(
    name="data_curator",
    root_agent=root_agent,
    plugins=[
        ReflectAndRetryToolPlugin(max_retries=3),
    ],
)
