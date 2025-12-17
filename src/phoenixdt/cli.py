"""PhoenixDT CLI module."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from phoenixdt.core.config import PhoenixConfig
from phoenixdt.core.digital_twin import DigitalTwin

# Setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file")
@click.option("--debug", is_flag=True, help="Debug mode")
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool):
    """PhoenixDT - Industrial Digital Twin."""

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        phoenix_config = PhoenixConfig.from_yaml(config) if config else PhoenixConfig()
        ctx.ensure_object(dict)
        ctx.obj["config"] = phoenix_config
        ctx.obj["debug"] = debug
    except Exception as e:
        console.print(f"[red]Config error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--duration", "-d", type=float, help="Run duration (seconds)")
@click.option("--speed", "-s", type=float, help="Target speed (RPM)")
@click.option("--load", "-l", type=float, help="Load torque (Nm)")
@click.pass_context
def run(ctx, duration: Optional[float], speed: Optional[float], load: Optional[float]):
    """Run digital twin simulation."""

    config = ctx.obj["config"]

    async def simulate():
        twin = DigitalTwin(config)

        # Set parameters
        if speed:
            twin.set_target_speed(speed)
        if load:
            twin.set_load_torque(load)

        console.print(
            Panel(
                f"[bold green]PhoenixDT Running[/bold green]\n"
                f"Target Speed: {twin.target_speed} RPM\n"
                f"Load Torque: {twin.load_torque} Nm"
                + (
                    f"\nDuration: {duration}s" if duration else "\nPress Ctrl+C to stop"
                ),
                title="Simulation",
            )
        )

        try:
            await twin.start(duration)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopped by user[/yellow]")
            await twin.stop()

    try:
        asyncio.run(simulate())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", type=int, default=8000, help="Server port")
@click.pass_context
def serve(ctx, host: str, port: int):
    """Start API server."""

    console.print(
        Panel(
            f"[bold green]PhoenixDT API Server[/bold green]\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"API Docs: http://{host}:{port}/docs\n"
            f"WebSocket: ws://{host}:{port}/ws",
            title="API Server",
        )
    )

    try:
        import uvicorn

        uvicorn.run(
            "phoenixdt.api.app:app", host=host, port=port, reload=ctx.obj["debug"]
        )
    except ImportError:
        console.print("[red]Install uvicorn: pip install uvicorn[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show system status."""

    config = ctx.obj["config"]

    async def show_status():
        twin = DigitalTwin(config)

        # Quick run to get status
        await twin.start()
        await asyncio.sleep(0.5)
        await twin.stop()

        status = twin.get_status()

        # Create status display
        status_text = Text()
        status_text.append(f"State: ", style="bold")
        status_text.append(f"{status['state']}\n", style="green")
        status_text.append(f"Time: ", style="bold")
        status_text.append(f"{status['simulation_time']:.2f}s\n\n")

        status_text.append("[bold]Motor:[/bold]\n")
        motor = status["motor"]
        status_text.append(f"Speed: {motor['speed']:.1f} RPM\n")
        status_text.append(f"Torque: {motor['torque']:.2f} Nm\n")
        status_text.append(f"Current: {motor['current']:.2f} A\n")
        status_text.append(f"Temperature: {motor['temperature']:.1f}°C\n")
        status_text.append(f"Power: {motor['power']:.2f} kW\n")
        status_text.append(f"Efficiency: {motor['efficiency']:.1f}%\n")

        status_text.append("\n[bold]Health:[/bold]\n")
        health = status["health"]
        status_text.append(f"Overall: {health['overall']:.2f}\n")
        status_text.append(f"Thermal: {health['thermal']:.2f}\n")
        status_text.append(f"Mechanical: {health['mechanical']:.2f}")

        console.print(Panel(status_text, title="System Status"))

    try:
        asyncio.run(show_status())
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def config(ctx):
    """Show configuration."""

    config = ctx.obj["config"]

    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Motor Power", f"{config.simulation.motor_power} kW")
    table.add_row("Motor Speed", f"{config.simulation.motor_speed} RPM")
    table.add_row("Load Torque", f"{config.simulation.load_torque} Nm")
    table.add_row("Time Step", f"{config.simulation.dt} s")
    table.add_row("Max Current", f"{config.control.safety_limits['max_current']} A")
    table.add_row(
        "Max Temperature", f"{config.control.safety_limits['max_temperature']}°C"
    )
    table.add_row("API Port", str(config.interface.api_port))

    console.print(table)


@cli.command()
@click.pass_context
def test(ctx):
    """Quick system test."""

    console.print("[bold blue]Running system test...[/bold blue]")

    async def run_test():
        config = ctx.obj["config"]
        twin = DigitalTwin(config)

        try:
            # Test initialization
            await twin.start()
            console.print("✓ Initialization", style="green")

            # Test control
            twin.set_target_speed(1500)
            await asyncio.sleep(1.0)
            console.print("✓ Control response", style="green")

            # Test status
            status = twin.get_status()
            if status["motor"]["speed"] > 0:
                console.print("✓ Motor simulation", style="green")

            # Test anomaly detection
            if status["health"]["overall"] > 0:
                console.print("✓ Health monitoring", style="green")

            await twin.stop()

            console.print("\n[bold green]✓ All tests passed![/bold green]")

        except Exception as e:
            console.print(f"[red]✗ Test failed: {e}[/red]")
            await twin.stop()
            sys.exit(1)

    try:
        asyncio.run(run_test())
    except Exception as e:
        console.print(f"[red]Test error: {e}[/red]")
        sys.exit(1)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
