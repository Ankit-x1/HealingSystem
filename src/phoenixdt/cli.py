"""
Command Line Interface for PhoenixDT

Clean, production-ready CLI with Rich interface.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from phoenixdt.core.config import PhoenixConfig
from phoenixdt.core.digital_twin import DigitalTwin


# Setup console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
)
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx, config: Optional[str], debug: bool):
    """PhoenixDT - Industrial Digital Twin CLI"""

    # Setup logging level
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    try:
        if config:
            phoenix_config = PhoenixConfig.from_yaml(config)
        else:
            phoenix_config = PhoenixConfig()

        # Store config in context
        ctx.ensure_object(dict)
        ctx.obj["config"] = phoenix_config
        ctx.obj["debug"] = debug

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--duration", "-d", type=float, help="Simulation duration in seconds")
@click.option("--speed", "-s", type=float, help="Target motor speed in RPM")
@click.option("--load", "-l", type=float, help="Load torque in Nm")
@click.pass_context
def start(
    ctx, duration: Optional[float], speed: Optional[float], load: Optional[float]
):
    """Start the digital twin simulation"""

    config = ctx.obj["config"]

    async def run_simulation():
        digital_twin = DigitalTwin(config)

        # Set initial parameters
        if speed:
            digital_twin.set_target_speed(speed)
        if load:
            digital_twin.set_load_torque(load)

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Starting simulation...", total=None)

                # Start digital twin
                await digital_twin.start()

                progress.update(task, description="Simulation running...")

                # Create live display
                with console.status("[bold green]Simulation running...") as status:
                    start_time = asyncio.get_event_loop().time()
                    last_update = start_time

                    while True:
                        current_time = asyncio.get_event_loop().time()

                        # Update display every second
                        if current_time - last_update >= 1.0:
                            status.update(
                                f"[bold green]Simulation running... ({current_time - start_time:.1f}s)"
                            )
                            last_update = current_time

                        # Check duration
                        if duration and (current_time - start_time) >= duration:
                            break

                        await asyncio.sleep(0.1)

                # Show final status
                await show_status(digital_twin)

        except KeyboardInterrupt:
            console.print("\n[yellow]Simulation interrupted by user[/yellow]")
        finally:
            await digital_twin.stop()

    try:
        asyncio.run(run_simulation())
    except Exception as e:
        console.print(f"[red]Simulation error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", type=int, help="Server port")
@click.pass_context
def serve(ctx, host: str, port: Optional[int]):
    """Start the API server"""

    config = ctx.obj["config"]

    # Override port if provided
    if port:
        config.interface.api_port = port

    console.print(
        Panel.fit(
            f"[bold green]PhoenixDT API Server[/bold green]\n\n"
            f"Host: {host}\n"
            f"Port: {config.interface.api_port}\n"
            f"Debug: {ctx.obj['debug']}\n\n"
            f"[blue]API Documentation: http://{host}:{config.interface.api_port}/docs[/blue]",
            title="Server Starting",
        )
    )

    try:
        import uvicorn

        uvicorn.run(
            "phoenixdt.api.app:app",
            host=host,
            port=config.interface.api_port,
            reload=ctx.obj["debug"],
            log_level="debug" if ctx.obj["debug"] else "info",
        )
    except ImportError:
        console.print(
            "[red]uvicorn not installed. Install with: pip install uvicorn[/red]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Show current system status"""

    config = ctx.obj["config"]

    async def show_system_status():
        digital_twin = DigitalTwin(config)

        try:
            # Start temporarily to get status
            await digital_twin.start()
            await asyncio.sleep(0.1)  # Brief initialization

            await show_status(digital_twin)

        finally:
            await digital_twin.stop()

    try:
        asyncio.run(show_system_status())
    except Exception as e:
        console.print(f"[red]Status error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def config_show(ctx):
    """Show current configuration"""

    config = ctx.obj["config"]

    # Create configuration table
    table = Table(title="PhoenixDT Configuration")
    table.add_column("Section", style="cyan")
    table.add_column("Parameter", style="magenta")
    table.add_column("Value", style="green")

    # Simulation config
    table.add_row("Simulation", "Time Step", f"{config.simulation.dt}s")
    table.add_row("", "Duration", f"{config.simulation.duration}s")
    table.add_row("", "Motor Power", f"{config.simulation.motor_power}kW")
    table.add_row("", "Motor Speed", f"{config.simulation.motor_speed}RPM")
    table.add_row("", "Load Torque", f"{config.simulation.load_torque}Nm")

    # ML config
    table.add_row("ML", "VAE Latent Dim", str(config.ml.vae_latent_dim))
    table.add_row("", "RL Algorithm", config.ml.rl_algorithm)
    table.add_row("", "Learning Rate", str(config.ml.rl_learning_rate))

    # Control config
    table.add_row("Control", "Frequency", f"{config.control.control_frequency}Hz")
    table.add_row("", "Max Current", f"{config.control.safety_limits['max_current']}A")
    table.add_row(
        "", "Max Temperature", f"{config.control.safety_limits['max_temperature']}°C"
    )

    # Interface config
    table.add_row("Interface", "API Port", str(config.interface.api_port))
    table.add_row("", "OPC-UA Port", str(config.interface.opcua_port))
    table.add_row("", "Dashboard Port", str(config.interface.dashboard_port))

    console.print(table)


@cli.command()
@click.argument("output_path", type=click.Path())
@click.pass_context
def config_export(ctx, output_path: str):
    """Export configuration to file"""

    config = ctx.obj["config"]

    try:
        output_path = Path(output_path)

        if output_path.suffix.lower() in [".yaml", ".yml"]:
            yaml_content = config.to_yaml(output_path)
            console.print(f"[green]Configuration exported to {output_path}[/green]")
        else:
            console.print(f"[red]Unsupported file format: {output_path.suffix}[/red]")
            console.print("Supported formats: .yaml, .yml")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Export error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.pass_context
def test(ctx):
    """Run system tests"""

    console.print("[bold blue]Running PhoenixDT system tests...[/bold blue]")

    async def run_tests():
        config = ctx.obj["config"]
        digital_twin = DigitalTwin(config)

        with Progress(console=console) as progress:
            # Test 1: Initialization
            task1 = progress.add_task("Testing initialization...", total=1)
            await digital_twin.start()
            progress.update(task1, advance=1)

            # Test 2: Control
            task2 = progress.add_task("Testing control...", total=1)
            digital_twin.set_target_speed(1500)
            await asyncio.sleep(0.5)
            progress.update(task2, advance=1)

            # Test 3: Status
            task3 = progress.add_task("Testing status...", total=1)
            status = digital_twin.get_status()
            progress.update(task3, advance=1)

            # Test 4: Anomaly detection
            task4 = progress.add_task("Testing anomaly detection...", total=1)
            # Simulate high temperature
            digital_twin.state.temperature = 120.0
            anomalies = digital_twin.analyzer.analyze_anomalies()
            progress.update(task4, advance=1)

            await digital_twin.stop()

        # Show results
        console.print("\n[bold green]✓ All tests passed![/bold green]")

        if anomalies:
            console.print(
                f"\n[yellow]Detected {len(anomalies)} anomalies during test:[/yellow]"
            )
            for anomaly in anomalies:
                console.print(f"  • {anomaly['type']}: {anomaly['severity']} severity")

    try:
        asyncio.run(run_tests())
    except Exception as e:
        console.print(f"[red]Test error: {e}[/red]")
        sys.exit(1)


async def show_status(digital_twin: DigitalTwin):
    """Display system status"""

    status = digital_twin.get_status()

    # Create status panel
    status_text = Text()
    status_text.append(f"State: ", style="bold")
    status_text.append(
        f"{status['state']}\n",
        style="green" if status["state"] == "running" else "yellow",
    )
    status_text.append(f"Simulation Time: ", style="bold")
    status_text.append(f"{status['simulation_time']:.2f}s\n")

    # Motor status
    motor = status["motor"]
    status_text.append("\n[bold]Motor Status:[/bold]\n")
    status_text.append(f"Speed: {motor['speed']:.1f} RPM\n")
    status_text.append(f"Torque: {motor['torque']:.2f} Nm\n")
    status_text.append(f"Current: {motor['current']:.2f} A\n")
    status_text.append(f"Temperature: {motor['temperature']:.1f}°C\n")
    status_text.append(f"Power: {motor['power']:.2f} kW\n")
    status_text.append(f"Efficiency: {motor['efficiency']:.1f}%\n")

    # Targets
    targets = status["targets"]
    status_text.append("\n[bold]Targets:[/bold]\n")
    status_text.append(f"Target Speed: {targets['speed']:.1f} RPM\n")
    status_text.append(f"Load Torque: {targets['load_torque']:.2f} Nm\n")

    console.print(Panel(status_text, title="System Status"))


if __name__ == "__main__":
    cli()
