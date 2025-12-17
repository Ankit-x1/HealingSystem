"""
Performance tests for PhoenixDT API using Locust
"""

from locust import HttpUser, task, between
import json
import random


class PhoenixDTUser(HttpUser):
    """Simulated user for PhoenixDT API performance testing"""

    wait_time = between(1, 3)

    def on_start(self):
        """Called when a user starts"""
        # Test basic connectivity
        self.client.get("/api/status")

    @task(3)
    def get_status(self):
        """Get system status - high frequency operation"""
        self.client.get("/api/status")

    @task(2)
    def get_state(self):
        """Get current digital twin state"""
        self.client.get("/api/state")

    @task(1)
    def get_performance(self):
        """Get performance analytics"""
        self.client.get("/api/performance")

    @task(1)
    def get_causal_analysis(self):
        """Get causal analysis"""
        self.client.get("/api/causal-analysis")

    @task(1)
    def inject_fault(self):
        """Inject a fault into the system"""
        fault_types = [
            "bearing_wear",
            "lubrication_loss",
            "overload",
            "voltage_unbalance",
        ]
        fault_data = {
            "fault_type": random.choice(fault_types),
            "severity": random.uniform(0.1, 0.8),
        }

        with self.client.post(
            "/api/fault", json=fault_data, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Fault injection failed: {response.text}")

    @task(1)
    def set_control(self):
        """Set control mode"""
        control_modes = ["rl", "pid", "manual"]
        control_data = {"control_mode": random.choice(control_modes)}

        if control_data["control_mode"] == "manual":
            control_data["manual_voltage"] = [
                random.uniform(-400, 400) for _ in range(3)
            ]

        with self.client.post(
            "/api/control", json=control_data, catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Control setting failed: {response.text}")


class AdminUser(HttpUser):
    """Admin user with system control operations"""

    wait_time = between(5, 10)
    weight = 1  # Less frequent than regular users

    @task
    def start_system(self):
        """Start the digital twin system"""
        with self.client.post("/api/start", catch_response=True) as response:
            if response.status_code in [200, 409]:  # 409 = already running
                response.success()
            else:
                response.failure(f"System start failed: {response.text}")

    @task
    def stop_system(self):
        """Stop the digital twin system"""
        with self.client.post("/api/stop", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"System stop failed: {response.text}")


class WebSocketUser(HttpUser):
    """User testing WebSocket connections"""

    wait_time = between(10, 30)
    weight = 1

    @task
    def test_websocket_connection(self):
        """Test WebSocket endpoint"""
        # Note: Locust doesn't natively support WebSocket testing
        # This would require custom implementation or using tools like Artillery
        pass
