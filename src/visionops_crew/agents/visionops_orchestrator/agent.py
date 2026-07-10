import os

from dotenv import load_dotenv


from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.google_search_agent_tool import (
    GoogleSearchAgentTool,
    create_google_search_agent,
)

from visionops_crew.agents.cv_research.agent import root_agent as cv_research
from visionops_crew.agents.data_curator.agent import root_agent as data_curator
from visionops_crew.agents.ml_engineer.agent import root_agent as ml_engineer
from visionops_crew.agents.vision_task_planner.agent import (
    root_agent as vision_task_planner,
)

from .prompts import (
    VISIONOPS_CREW_ORCHESTRATOR_INSTRUCTION,
)
from .tools import run_antigravity_agent

load_dotenv(".env")
DEFAULT_MODEL = "gemini-3.5-flash"


search_agent = create_google_search_agent(
    model=os.getenv(
        "GOOGLE_SEARCH_MODEL",
        os.getenv("VISIONOPS_ORCHESTRATOR_MODEL", DEFAULT_MODEL),
    )
)
root_agent = Agent(
    model=os.getenv("VISIONOPS_ORCHESTRATOR_MODEL", DEFAULT_MODEL),
    name="visionops_orchestrator",
    description=(
        "A computer vision expert orchestrator that uses a planning specialist, "
        "Google Search, computer vision research, data curation, dataset "
        "operations, ML engineering support, and opt-in Antigravity super-agent "
        "support for robust CV workflows."
    ),
    instruction=VISIONOPS_CREW_ORCHESTRATOR_INSTRUCTION,
    tools=[
        run_antigravity_agent,
        AgentTool(agent=vision_task_planner),
        GoogleSearchAgentTool(agent=search_agent),
        AgentTool(agent=cv_research),
        AgentTool(agent=ml_engineer),
        AgentTool(agent=data_curator),
    ],
)

app = App(
    name="visionops_orchestrator",
    root_agent=root_agent,
    plugins=[
        ReflectAndRetryToolPlugin(max_retries=3),
    ],
)
