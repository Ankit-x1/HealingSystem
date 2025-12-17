"""
Main entry point for PhoenixDT application
"""

import asyncio
import argparse
import logging
from pathlib import Path
from typing import Optional

from phoenixdt.core.config import Config
from phoenixdt.core.digital_twin import DigitalTwin
from phoenixdt.interfaces.opcua_server import OpcUaServer
from phoenixdt.dashboard.app import PhoenixDTDashboard


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


async def run_digital_twin(config: Config, duration: Optional[float] = None):
    """Run digital twin with OPC-UA server"""
    digital_twin = DigitalTwin(config)
    opcua_server = OpcUaServer(config.interface, digital_twin)

    try:
        # Start OPC-UA server
        opcua_task = asyncio.create_task(opcua_server.start())

        # Start digital twin
        await digital_twin.start(duration)

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await opcua_server.stop()
        digital_twin.stop()


def run_dashboard():
    """Run Streamlit dashboard"""
    import streamlit.web.cli as stcli
    import sys

    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    sys.argv = ["streamlit", "run", str(dashboard_path), "--server.port=8501"]
    sys.exit(stcli.main())


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PhoenixDT - Failure-Aware Digital Twin"
    )
    parser.add_argument(
        "--mode",
        choices=["twin", "dashboard", "api"],
        default="twin",
        help="Running mode",
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--duration", type=float, help="Simulation duration in seconds")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration
    if args.config:
        config = Config.from_yaml(Path(args.config))
    else:
        config = Config()

    if args.mode == "twin":
        asyncio.run(run_digital_twin(config, args.duration))
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "api":
        # TODO: Implement API server
        print("API server mode not yet implemented")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
