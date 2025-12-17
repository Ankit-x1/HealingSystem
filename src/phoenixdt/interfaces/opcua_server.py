"""
OPC-UA Server Interface for Industrial Communication

Provides OPC-UA server functionality to expose digital twin data
to industrial clients and enable real-time monitoring and control.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
from loguru import logger

try:
    from asyncua import Server, ua
    from asyncua.common.methods import uamethod

    ASYNCUA_AVAILABLE = True
except ImportError:
    ASYNCUA_AVAILABLE = False
    logger.warning("asyncua not available, OPC-UA functionality disabled")

from ..core.config import InterfaceConfig
from ..core.digital_twin import DigitalTwin, TwinState


@dataclass
class OPCUANode:
    """OPC-UA node definition"""

    node_id: str
    browse_name: str
    display_name: str
    data_type: type
    value: Any
    description: str = ""
    access_level: int = 3  # Read + Write


class OpcUaServer:
    """OPC-UA server for digital twin interface"""

    def __init__(self, config: InterfaceConfig, digital_twin: DigitalTwin):
        if not ASYNCUA_AVAILABLE:
            raise ImportError("asyncua is required for OPC-UA functionality")

        self.config = config
        self.digital_twin = digital_twin
        self.server = None
        self.is_running = False

        # Node storage
        self.nodes: Dict[str, OPCUANode] = {}
        self.namespace_index = None

        # Callbacks
        self.control_callbacks: Dict[str, Callable] = {}

        logger.info("OPC-UA server initialized")

    async def start(self):
        """Start OPC-UA server"""
        try:
            # Create server instance
            self.server = Server()

            # Configure server endpoint
            server_url = f"opc.tcp://0.0.0.0:{self.config.opcua_port}/phoenixdt/server/"
            await self.server.init()
            self.server.set_endpoint(server_url)

            # Set server info
            self.server.set_server_name("PhoenixDT Digital Twin Server")
            self.server.set_application_uri("urn:AnkitKarki:PhoenixDT:Server")

            # Setup namespace
            self.namespace_index = await self.server.register_namespace(
                "http://github.com/Ankit-x1/HealingSystem/PhoenixDT"
            )

            # Create address space
            await self._create_address_space()

            # Start server
            async with self.server:
                self.is_running = True
                logger.info(f"OPC-UA server started at {server_url}")

                # Keep server running
                while self.is_running:
                    await asyncio.sleep(1)
                    await self._update_node_values()

        except Exception as e:
            logger.error(f"Failed to start OPC-UA server: {e}")
            raise

    async def _create_address_space(self):
        """Create OPC-UA address space structure"""
        # Get objects folder
        objects = self.server.get_objects_node()

        # Create PhoenixDT folder
        phoenixdt_folder = await objects.add_folder(self.namespace_index, "PhoenixDT")

        # Create main categories
        motor_folder = await phoenixdt_folder.add_folder(self.namespace_index, "Motor")

        bearing_folder = await phoenixdt_folder.add_folder(
            self.namespace_index, "Bearing"
        )

        control_folder = await phoenixdt_folder.add_folder(
            self.namespace_index, "Control"
        )

        health_folder = await phoenixdt_folder.add_folder(
            self.namespace_index, "Health"
        )

        # Motor variables
        motor_nodes = [
            OPCUANode(
                node_id="Motor_Speed",
                browse_name="Speed",
                display_name="Motor Speed (RPM)",
                data_type=float,
                value=0.0,
                description="Current motor speed in revolutions per minute",
            ),
            OPCUANode(
                node_id="Motor_Torque",
                browse_name="Torque",
                display_name="Motor Torque (Nm)",
                data_type=float,
                value=0.0,
                description="Current motor torque in Newton-meters",
            ),
            OPCUANode(
                node_id="Motor_Current",
                browse_name="Current",
                display_name="Motor Current (A)",
                data_type=float,
                value=0.0,
                description="Current motor current in Amperes",
            ),
            OPCUANode(
                node_id="Motor_Voltage",
                browse_name="Voltage",
                display_name="Motor Voltage (V)",
                data_type=float,
                value=0.0,
                description="Current motor voltage in Volts",
            ),
            OPCUANode(
                node_id="Motor_Power",
                browse_name="Power",
                display_name="Motor Power (W)",
                data_type=float,
                value=0.0,
                description="Current motor power consumption in Watts",
            ),
            OPCUANode(
                node_id="Motor_Efficiency",
                browse_name="Efficiency",
                display_name="Motor Efficiency (%)",
                data_type=float,
                value=0.0,
                description="Current motor efficiency as percentage",
            ),
        ]

        # Bearing variables
        bearing_nodes = [
            OPCUANode(
                node_id="Bearing_Wear",
                browse_name="Wear",
                display_name="Bearing Wear Level",
                data_type=float,
                value=0.0,
                description="Current bearing wear level (0-1)",
            ),
            OPCUANode(
                node_id="Bearing_Temperature",
                browse_name="Temperature",
                display_name="Bearing Temperature (°C)",
                data_type=float,
                value=25.0,
                description="Current bearing temperature in Celsius",
            ),
            OPCUANode(
                node_id="Bearing_Vibration",
                browse_name="Vibration",
                display_name="Bearing Vibration (mm/s)",
                data_type=float,
                value=0.0,
                description="Current bearing vibration in mm/s",
            ),
            OPCUANode(
                node_id="Bearing_Lubrication",
                browse_name="Lubrication",
                display_name="Lubrication Quality",
                data_type=float,
                value=1.0,
                description="Current lubrication quality (0-1)",
            ),
        ]

        # Control variables
        control_nodes = [
            OPCUANode(
                node_id="Control_Mode",
                browse_name="Mode",
                display_name="Control Mode",
                data_type=str,
                value="rl",
                description="Current control mode (rl/pid/manual)",
                access_level=3,
            ),
            OPCUANode(
                node_id="Control_Voltage_A",
                browse_name="VoltageA",
                display_name="Phase A Voltage (V)",
                data_type=float,
                value=0.0,
                description="Phase A voltage command",
                access_level=3,
            ),
            OPCUANode(
                node_id="Control_Voltage_B",
                browse_name="VoltageB",
                display_name="Phase B Voltage (V)",
                data_type=float,
                value=0.0,
                description="Phase B voltage command",
                access_level=3,
            ),
            OPCUANode(
                node_id="Control_Voltage_C",
                browse_name="VoltageC",
                display_name="Phase C Voltage (V)",
                data_type=float,
                value=0.0,
                description="Phase C voltage command",
                access_level=3,
            ),
        ]

        # Health variables
        health_nodes = [
            OPCUANode(
                node_id="Health_Overall",
                browse_name="Overall",
                display_name="Overall Health Score",
                data_type=float,
                value=1.0,
                description="Overall system health score (0-1)",
            ),
            OPCUANode(
                node_id="Health_RUL",
                browse_name="RUL",
                display_name="Remaining Useful Life (hours)",
                data_type=float,
                value=8760.0,
                description="Estimated remaining useful life in hours",
            ),
            OPCUANode(
                node_id="Health_Anomaly_Count",
                browse_name="AnomalyCount",
                display_name="Active Anomalies",
                data_type=int,
                value=0,
                description="Number of active anomalies",
            ),
        ]

        # Create OPC-UA nodes
        for node in motor_nodes:
            await self._create_opcua_node(motor_folder, node)

        for node in bearing_nodes:
            await self._create_opcua_node(bearing_folder, node)

        for node in control_nodes:
            await self._create_opcua_node(control_folder, node)

        for node in health_nodes:
            await self._create_opcua_node(health_folder, node)

        # Add methods
        await self._add_methods(phoenixdt_folder)

        logger.info("OPC-UA address space created")

    async def _create_opcua_node(self, parent_folder, node: OPCUANode):
        """Create individual OPC-UA node"""
        node_id = ua.NodeId(node.node_id, self.namespace_index)
        browse_name = ua.QualifiedName(node.browse_name, self.namespace_index)

        opcua_node = await parent_folder.add_variable(
            node_id=node_id, bname=browse_name, val=node.value, vtype=node.data_type
        )

        await opcua_node.set_display_name(ua.LocalizedText(node.display_name))
        await opcua_node.set_description(ua.LocalizedText(node.description))

        # Set access level
        await opcua_node.set_access_level(node.access_level)

        # Store reference
        self.nodes[node.node_id] = node

        # Add write callback for writable nodes
        if node.access_level >= 2:  # Write access
            await opcua_node.set_writable()

    async def _add_methods(self, parent_folder):
        """Add OPC-UA methods"""

        @uamethod
        async def reset_system(parent):
            """Reset the digital twin system"""
            self.digital_twin.reset()
            logger.info("System reset via OPC-UA")
            return True

        @uamethod
        async def inject_fault(parent, fault_type: str, severity: float):
            """Inject a fault into the system"""
            try:
                self.digital_twin.simulator.inject_fault(fault_type, severity)
                logger.info(
                    f"Fault injected via OPC-UA: {fault_type}, severity: {severity}"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to inject fault: {e}")
                return False

        @uamethod
        async def set_control_mode(parent, mode: str):
            """Set control mode"""
            try:
                self.digital_twin.set_control_mode(mode)
                logger.info(f"Control mode set via OPC-UA: {mode}")
                return True
            except Exception as e:
                logger.error(f"Failed to set control mode: {e}")
                return False

        # Add methods to server
        await parent_folder.add_method(
            self.namespace_index, "ResetSystem", reset_system, [], [ua.Boolean]
        )

        await parent_folder.add_method(
            self.namespace_index,
            "InjectFault",
            inject_fault,
            [ua.String, ua.Double],
            [ua.Boolean],
        )

        await parent_folder.add_method(
            self.namespace_index,
            "SetControlMode",
            set_control_mode,
            [ua.String],
            [ua.Boolean],
        )

    async def _update_node_values(self):
        """Update node values from digital twin state"""
        current_state = self.digital_twin.get_current_state()
        if not current_state:
            return

        # Update motor nodes
        physical = current_state.physical_state

        updates = {
            "Motor_Speed": physical.get("speed_rpm", 0.0),
            "Motor_Torque": physical.get("torque_nm", 0.0),
            "Motor_Current": physical.get("current_a", 0.0),
            "Motor_Voltage": physical.get("voltage_v", 0.0),
            "Motor_Power": physical.get("power_w", 0.0),
            "Motor_Efficiency": physical.get("efficiency", 0.0) * 100,
            "Bearing_Wear": physical.get("bearing_wear", 0.0),
            "Bearing_Temperature": physical.get("bearing_temp", 25.0),
            "Bearing_Vibration": physical.get("vibration_mm_s", 0.0),
            "Bearing_Lubrication": self.digital_twin.simulator.bearing_state.lubrication_quality,
            "Health_Overall": current_state.health_metrics.get("overall_health", 1.0),
            "Health_RUL": current_state.health_metrics.get(
                "remaining_useful_life_hours", 8760.0
            ),
            "Health_Anomaly_Count": len(current_state.anomalies),
            "Control_Mode": self.digital_twin.control_mode,
        }

        # Update control voltages
        if current_state.control_actions and "voltage" in current_state.control_actions:
            voltages = current_state.control_actions["voltage"]
            if len(voltages) >= 3:
                updates["Control_Voltage_A"] = voltages[0]
                updates["Control_Voltage_B"] = voltages[1]
                updates["Control_Voltage_C"] = voltages[2]

        # Apply updates
        for node_id, value in updates.items():
            if node_id in self.nodes:
                try:
                    node = await self.server.get_node(
                        ua.NodeId(node_id, self.namespace_index)
                    )
                    await node.write_value(ua.DataValue(ua.Variant(value)))
                except Exception as e:
                    logger.warning(f"Failed to update node {node_id}: {e}")

    async def stop(self):
        """Stop OPC-UA server"""
        self.is_running = False
        if self.server:
            await self.server.stop()
        logger.info("OPC-UA server stopped")

    def get_endpoint_url(self) -> str:
        """Get OPC-UA server endpoint URL"""
        return f"opc.tcp://localhost:{self.config.opcua_port}/phoenixdt/server/"

    def get_namespace_uri(self) -> str:
        """Get namespace URI"""
        return "http://github.com/Ankit-x1/HealingSystem/PhoenixDT"

    async def write_node_value(self, node_id: str, value: Any):
        """Write value to a specific node"""
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node ID: {node_id}")

        node = self.nodes[node_id]
        if node.access_level < 2:  # No write access
            raise PermissionError(f"Node {node_id} is not writable")

        try:
            opcua_node = await self.server.get_node(
                ua.NodeId(node_id, self.namespace_index)
            )
            await opcua_node.write_value(ua.DataValue(ua.Variant(value)))

            # Handle control actions
            if node_id.startswith("Control_"):
                await self._handle_control_action(node_id, value)

        except Exception as e:
            logger.error(f"Failed to write node {node_id}: {e}")
            raise

    async def _handle_control_action(self, node_id: str, value: Any):
        """Handle control actions from OPC-UA writes"""
        if node_id == "Control_Mode":
            self.digital_twin.set_control_mode(str(value))

        elif node_id in ["Control_Voltage_A", "Control_Voltage_B", "Control_Voltage_C"]:
            # Update manual control voltages
            current_voltages = self.digital_twin.manual_control.copy()

            if node_id == "Control_Voltage_A":
                current_voltages[0] = float(value)
            elif node_id == "Control_Voltage_B":
                current_voltages[1] = float(value)
            elif node_id == "Control_Voltage_C":
                current_voltages[2] = float(value)

            self.digital_twin.set_manual_control(current_voltages)

            # Switch to manual mode if not already
            if self.digital_twin.control_mode != "manual":
                self.digital_twin.set_control_mode("manual")

    def get_node_info(self) -> Dict[str, Dict]:
        """Get information about all nodes"""
        return {
            node_id: {
                "browse_name": node.browse_name,
                "display_name": node.display_name,
                "data_type": node.data_type.__name__,
                "description": node.description,
                "access_level": node.access_level,
                "current_value": node.value,
            }
            for node_id, node in self.nodes.items()
        }

    async def export_nodeset(self, filepath: str):
        """Export nodeset to XML file"""
        # Create XML structure
        root = ET.Element("UANodeSet")
        root.set("xmlns", "http://opcfoundation.org/UA/2011/03/UANodeSet.xsd")

        # Add namespace
        namespace = ET.SubElement(root, "NamespaceUris")
        ET.SubElement(namespace, "Uri").text = self.get_namespace_uri()

        # Add nodes
        for node_id, node in self.nodes.items():
            ua_node = ET.SubElement(root, "UAVariable")
            ua_node.set("NodeId", f"ns={self.namespace_index};s={node_id}")
            ua_node.set("BrowseName", f"{self.namespace_index}:{node.browse_name}")
            ua_node.set("DataType", node.data_type.__name__)

            display_name = ET.SubElement(ua_node, "DisplayName")
            display_name.text = node.display_name

            description = ET.SubElement(ua_node, "Description")
            description.text = node.description

        # Write to file
        tree = ET.ElementTree(root)
        tree.write(filepath, encoding="utf-8", xml_declaration=True)

        logger.info(f"Nodeset exported to {filepath}")


class OpcUaClient:
    """OPC-UA client for testing and integration"""

    def __init__(self, endpoint_url: str):
        if not ASYNCUA_AVAILABLE:
            raise ImportError("asyncua is required for OPC-UA functionality")

        self.endpoint_url = endpoint_url
        self.client = None
        self.is_connected = False

    async def connect(self):
        """Connect to OPC-UA server"""
        try:
            self.client = Client(url=self.endpoint_url)
            await self.client.connect()
            self.is_connected = True
            logger.info(f"Connected to OPC-UA server at {self.endpoint_url}")
        except Exception as e:
            logger.error(f"Failed to connect to OPC-UA server: {e}")
            raise

    async def disconnect(self):
        """Disconnect from OPC-UA server"""
        if self.client:
            await self.client.disconnect()
            self.is_connected = False
            logger.info("Disconnected from OPC-UA server")

    async def read_node(self, node_id: str, namespace_index: int = 2):
        """Read value from a node"""
        if not self.is_connected:
            raise RuntimeError("Not connected to OPC-UA server")

        try:
            node = self.client.get_node(f"ns={namespace_index};s={node_id}")
            value = await node.read_value()
            return value
        except Exception as e:
            logger.error(f"Failed to read node {node_id}: {e}")
            raise

    async def write_node(self, node_id: str, value: Any, namespace_index: int = 2):
        """Write value to a node"""
        if not self.is_connected:
            raise RuntimeError("Not connected to OPC-UA server")

        try:
            node = self.client.get_node(f"ns={namespace_index};s={node_id}")
            await node.write_value(value)
            logger.info(f"Wrote value {value} to node {node_id}")
        except Exception as e:
            logger.error(f"Failed to write node {node_id}: {e}")
            raise

    async def call_method(self, method_name: str, *args, namespace_index: int = 2):
        """Call a method on the server"""
        if not self.is_connected:
            raise RuntimeError("Not connected to OPC-UA server")

        try:
            # Get objects folder
            objects = self.client.get_objects_node()

            # Get PhoenixDT folder
            phoenixdt = await objects.get_child([f"{namespace_index}:PhoenixDT"])

            # Call method
            result = await phoenixdt.call_method(
                f"{namespace_index}:{method_name}", *args
            )
            return result
        except Exception as e:
            logger.error(f"Failed to call method {method_name}: {e}")
            raise

    async def browse_nodes(
        self, parent_node_id: str = "i=85", namespace_index: int = 2
    ):
        """Browse nodes from a parent node"""
        if not self.is_connected:
            raise RuntimeError("Not connected to OPC-UA server")

        try:
            parent = self.client.get_node(parent_node_id)
            children = await parent.get_children()

            node_info = []
            for child in children:
                browse_name = await child.read_browse_name()
                display_name = await child.read_display_name()
                node_class = await child.read_node_class()

                node_info.append(
                    {
                        "node_id": str(child.nodeid),
                        "browse_name": str(browse_name),
                        "display_name": str(display_name.Text),
                        "node_class": str(node_class),
                    }
                )

            return node_info
        except Exception as e:
            logger.error(f"Failed to browse nodes: {e}")
            raise
