---
name: jupyter-notebook
description: Use Jupyter MCP tools to create, edit, execute, and manage notebooks.
metadata:
  adk_additional_tools:
    - use_notebook
    - list_notebooks
    - read_notebook
    - insert_cell
    - overwrite_cell_source
    - edit_cell_source
    - execute_cell
    - insert_execute_code_cell
    - read_cell
    - delete_cell
    - move_cell
    - execute_code
    - list_files
    - list_kernels
    - restart_notebook
    - unuse_notebook
    - connect_to_jupyter
---

# Jupyter Notebook Skill

## Purpose

Use this skill whenever the user asks to work with Jupyter notebooks, Python, data analysis, machine learning, visualization, scientific computing, or exploratory coding.

Always prefer the available Jupyter MCP tools over local Python execution when notebook-based work is appropriate.

## Workflow

1. Inspect existing notebooks.
2. Reuse an existing notebook when appropriate.
3. Otherwise create a new notebook.
4. Add Markdown explanations before each major section.
5. Insert code cells incrementally. Do NOT execute them unless the user explicitly indicates that you should do so.
6. Save the notebook frequently.
7. Summarize the notebook path and changes.

## Preferred MCP tools

### Notebook management

- use_notebook
- list_notebooks
- read_notebook

### Cell operations

- insert_markdown_cell
- insert_code_cell
- insert_execute_code_cell
- execute_cell
- read_cell
- overwrite_cell_source
- delete_cell

### Execution

Use execute_code only for temporary execution that should not become part of the notebook.

Persist all meaningful work inside notebook cells.

## Notebook style

Structure notebooks with:

1. Title
2. Objective
3. Imports
4. Load data
5. Exploration
6. Analysis
7. Conclusions

Use one logical step per code cell.

Add Markdown cells before each major section.

## Best practices

- Prefer notebook structure for interactive prototyping and exploration.
- Do NOT execute notebook cells automatically (e.g. using execute_cell, execute_code, or insert_execute_code_cell) unless the user explicitly indicates that they want you to do so.
- Keep notebooks reproducible.
- Keep Markdown synchronized with the code.
- Save after major milestones.
- Ask before deleting notebooks or overwriting user work.

## Completion

Before finishing:

- Ensure all cells are structured correctly, clean, and well-commented.
- Save all changes.
- Summarize what was created or modified.
- Mention any generated code structures or files.
