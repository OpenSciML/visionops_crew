import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

from .prompts import (
    CV_RESEARCH_AGENT_INSTRUCTION,
)

load_dotenv(".env")

hf_headers = {}
if hf_token := os.getenv("HF_TOKEN"):
    hf_headers["Authorization"] = f"Bearer {hf_token}"

hf_mcp_tools = McpToolset(
    connection_params=StreamableHTTPConnectionParams(
        url="https://huggingface.co/mcp",
        headers=hf_headers,
    )
)

root_agent = Agent(
    name="cv_research",
    model=os.getenv(
        "CV_RESEARCH_MODEL",
        "gemini-3.5-flash"
    ),
    description=(
        "A computer vision research specialist that can find and compare "
        "models, datasets, papers, documentation, benchmarks, and licensing "
        "metadata using the Hugging Face MCP server."
    ),
    instruction=CV_RESEARCH_AGENT_INSTRUCTION,
    tools=[hf_mcp_tools],
)

app = App(
    name="cv_research",
    root_agent=root_agent,
    plugins=[
        ReflectAndRetryToolPlugin(max_retries=3),
    ],
)
