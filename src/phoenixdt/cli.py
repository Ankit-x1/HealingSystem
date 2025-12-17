"""
PhoenixDT Command Line Interface

Production-ready CLI for industrial digital twin.
"""

from __future__ import annotations
import asyncio
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from loguru import logger

from .core.config import PhoenixConfig
from .core.digital_twin import DigitalTwin
from .api.app import create_app, run_server

# Create rich console
console = Console()

# Create CLI app
app = typer.Typer(
    name="phoenixdt",
    help="PhoenixDT - Industrial Digital Twin",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.command()
def start(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file path",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    duration: Optional[float] = typer.Option(
        None, "--duration", "-d", help="Simulation duration in seconds"
    ),
):
    """Start digital twin simulation"""
    try:
        # Load configuration
        if config:
            phoenix_config = PhoenixConfig.from_yaml(config)
            console.print(f"Loaded configuration from {config}")
        else:
            phoenix_config = PhoenixConfig()
            console.print("Using default configuration")

        # Initialize and start digital twin
        digital_twin = DigitalTwin(phoenix_config)

        if duration:
            console.print(f"Starting simulation for {duration} seconds")
            asyncio.run(digital_twin.start(duration=duration))
        else:
            console.print("Starting simulation (press Ctrl+C to stop)")
            asyncio.run(digital_twin.start())

    except Exception as e:
        console.print(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
):
    """Start API server"""
    try:
        # Load configuration
        if config:
            phoenix_config = PhoenixConfig.from_yaml(config)
        else:
            phoenix_config = PhoenixConfig()

        # Start server
        app = create_app()
        console.print(f"Starting server on {host}:{port}")
        run_server(host=host, port=port)

    except Exception as e:
        console.print(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """Show system status"""
    console.print("PhoenixDT Industrial Digital Twin")
    console.print("Status: Ready")
    console.print("Version: 2.0.0")


@app.command()
def config(
    action: str = typer.Argument(..., help="Action: show, create-sample"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
):
    """Configuration management"""
    if action == "show":
        # Show default configuration
        config = PhoenixConfig()
        console.print(config.dict())
    elif action == "create-sample":
        # Create sample configuration
        config = PhoenixConfig()
        sample_config = config.dict()

        if output:
            import yaml

            with open(output, "w") as f:
                yaml.dump(sample_config, f)
            console.print(f"Sample configuration created at {output}")
        else:
            console.print(sample_config)
    else:
        console.print(f"Unknown action: {action}")


def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
