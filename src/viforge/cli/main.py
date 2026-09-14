"""
ViForge Production CLI: End-to-End Specialization & Pareto Evaluation Platform.
Modular CLI entrypoint aggregating commands across domain sub-modules.
"""

import typer

from viforge.cli.commands import (
    data,
    evaluate,
    export,
    orchestrate,
    profile,
    train,
    ui,
)
from viforge.cli.console import console

app = typer.Typer(
    name="viforge",
    help="ViForge: Forging Small Models into Specialists — Production Experimentation Platform.",
    add_completion=False,
)

# Register modular CLI command suites
profile.register(app)
data.register(app)
train.register(app)
evaluate.register(app)
export.register(app)
orchestrate.register(app)
ui.register(app)

__all__ = ["app", "console"]

if __name__ == "__main__":
    app()
