"""
Interactive Studio / Streamlit dashboard launcher CLI command.
"""

import shutil
import subprocess
import typer
from rich.panel import Panel

from viforge.cli.console import console


def register(app: typer.Typer) -> None:
    """Register UI launcher command."""

    @app.command("ui")
    def cli_ui(
        port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit dashboard"),
        host: str = typer.Option("localhost", "--host", "-h", help="Host address for Streamlit dashboard"),
        browser: bool = typer.Option(True, "--browser/--no-browser", help="Automatically launch browser"),
    ):
        """Launch the ViForge Interactive Studio (Streamlit Dashboard)."""
        from viforge.ui import DASHBOARD_APP_PATH

        console.print(Panel.fit("[bold cyan]ViForge Interactive Studio Launcher[/bold cyan]"))

        streamlit_bin = shutil.which("streamlit")
        if not streamlit_bin:
            console.print(
                "[bold red]Streamlit is not installed.[/bold red]\n"
                "Install it via: [green]pip install \"viforge[ui]\"[/green] or [green]pip install streamlit plotly[/green]"
            )
            raise typer.Exit(code=1)

        cmd = [
            streamlit_bin,
            "run",
            str(DASHBOARD_APP_PATH),
            "--server.port",
            str(port),
            "--server.address",
            str(host),
        ]
        if not browser:
            cmd.append("--server.headless=true")

        console.print(f"Launching dashboard at: [bold green]http://{host}:{port}[/bold green]")
        console.print("[dim]Press Ctrl+C to stop the dashboard server.[/dim]")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            console.print("\n[bold yellow]ViForge Studio stopped.[/bold yellow]")
