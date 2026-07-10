import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.plugins import ReflectAndRetryToolPlugin
from google.adk.planners import BuiltInPlanner
from google.genai import types
from .prompts import (
    VISION_TASK_PLANNER_INSTRUCTION,
)

load_dotenv(".env")

root_agent = Agent(
    model=os.getenv(
        "VISION_TASK_PLANNER_MODEL",
        os.getenv("VISIONOPS_ORCHESTRATOR_MODEL", "gemini-3.5-flash"),
    ),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        )
    ),
    name="vision_task_planner",
    description=(
        "A computer vision planning specialist that clarifies user goals, "
        "classifies the CV task, identifies missing constraints, and proposes "
        "a practical CV pipeline plan."
    ),
    instruction=VISION_TASK_PLANNER_INSTRUCTION,
)

app = App(
    name="vision_task_planner",
    root_agent=root_agent,
    plugins=[
        ReflectAndRetryToolPlugin(max_retries=3),
    ],
)
