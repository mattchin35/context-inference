context-inference
==============================

Analysis of mouse behavior, electrophysiology, and computational modeling for a context-dependent reward seeking task.
Project structure loosely organized by Cookiecutter Data Science project template, but it could use an overhaul before 
being shared publicly.

Project Organization
------------
TODO


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Using Mouse Behavior Analysis code


# Using the Model Agents

Model-agent runs are controlled through the behavior-modeling controller. A run can be generated on its own, saved to
disk, plotted from the output dataframe, or both saved and plotted in the same session.

Plotting is handled separately from run saving. The controller supports:
- generating a run without saving it
- saving run outputs without plotting
- generating and saving a plot
- showing a plot without writing it to disk

Controller plots are built from the output dataframe and are intended to show actions, rewards, and model values
together on a single figure so the model trajectory can be inspected alongside the observed simulated behavior.

## Running tests

This repository currently supports test execution through `uv` from the repository root:

```bash
uv run pytest
```

Pytest collection is intentionally scoped to `src/tests` in the initial packaging cleanup pass.


