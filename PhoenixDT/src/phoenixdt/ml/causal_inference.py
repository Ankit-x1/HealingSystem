"""
Causal Inference Module using DoWhy

Provides explainable AI by identifying causal relationships
between system parameters and failures, enabling root cause analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

try:
    import dowhy
    from dowhy import CausalModel

    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger.warning("DoWhy not available, using simplified causal inference")


@dataclass
class CausalEffect:
    """Result of causal inference analysis"""

    treatment: str
    outcome: str
    ate: float  # Average Treatment Effect
    confidence_interval: Tuple[float, float]
    p_value: float
    method: str
    assumptions: List[str]


@dataclass
class CausalGraph:
    """Causal relationship graph"""

    nodes: List[str]
    edges: List[Tuple[str, str]]
    edge_weights: Dict[Tuple[str, str], float]


class CausalInference:
    """Causal inference engine for explainable AI"""

    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        self.causal_graph: Optional[CausalGraph] = None
        self.causal_effects: Dict[str, CausalEffect] = {}

        # Parameter definitions
        self.parameters = [
            "speed",
            "torque",
            "current",
            "voltage",
            "bearing_wear",
            "temperature",
            "vibration",
            "lubrication",
        ]

        # Failure indicators
        self.failure_indicators = [
            "high_vibration",
            "overheating",
            "bearing_failure",
            "efficiency_drop",
            "unstable_speed",
        ]

        logger.info("Causal inference module initialized")

    def fit(self, data: pd.DataFrame):
        """Fit causal model to data"""
        logger.info(f"Fitting causal model to {len(data)} samples")

        self.data = data.copy()

        # Preprocess data
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns] = self.scaler.fit_transform(
            self.data[numeric_columns]
        )

        # Create failure indicators
        self._create_failure_indicators()

        # Learn causal structure
        self._learn_causal_structure()

        # Estimate causal effects
        self._estimate_causal_effects()

        logger.info("Causal model fitting completed")

    def _create_failure_indicators(self):
        """Create binary failure indicators from continuous variables"""
        if self.data is None:
            return

        # High vibration (> 95th percentile)
        vibration_threshold = np.percentile(self.data.get("vibration_mm_s", []), 95)
        self.data["high_vibration"] = (
            self.data.get("vibration_mm_s", 0) > vibration_threshold
        ).astype(int)

        # Overheating (> 90°C)
        self.data["overheating"] = (self.data.get("bearing_temp", 0) > 90).astype(int)

        # Bearing failure (wear > 80%)
        self.data["bearing_failure"] = (self.data.get("bearing_wear", 0) > 0.8).astype(
            int
        )

        # Efficiency drop (< 70%)
        self.data["efficiency_drop"] = (self.data.get("efficiency", 1) < 0.7).astype(
            int
        )

        # Unstable speed (high variance in recent samples)
        if "speed_rpm" in self.data.columns:
            speed_std = self.data["speed_rpm"].rolling(window=10).std()
            speed_threshold = np.percentile(speed_std.dropna(), 90)
            self.data["unstable_speed"] = (speed_std > speed_threshold).astype(int)

    def _learn_causal_structure(self):
        """Learn causal graph structure from data"""
        if self.data is None:
            return

        # Create domain knowledge-based causal graph
        # This is a simplified approach - in practice, you'd use
        # structure learning algorithms or domain expertise

        nodes = self.parameters + self.failure_indicators
        edges = []
        edge_weights = {}

        # Physical causal relationships
        causal_rules = {
            # Mechanical relationships
            ("bearing_wear", "vibration_mm_s"): 0.8,
            ("bearing_wear", "temperature"): 0.6,
            ("vibration_mm_s", "high_vibration"): 0.9,
            ("temperature", "overheating"): 0.9,
            ("bearing_wear", "bearing_failure"): 0.7,
            # Electrical relationships
            ("current_a", "temperature"): 0.5,
            ("voltage_v", "current_a"): 0.6,
            ("torque_nm", "current_a"): 0.7,
            # Performance relationships
            ("temperature", "efficiency"): -0.4,
            ("vibration_mm_s", "efficiency"): -0.3,
            ("bearing_wear", "efficiency"): -0.6,
            ("efficiency", "efficiency_drop"): -0.8,
            # Speed stability
            ("torque_nm", "speed_rpm"): 0.5,
            ("voltage_v", "speed_rpm"): 0.4,
            ("bearing_wear", "speed_rpm"): -0.3,
            ("speed_rpm", "unstable_speed"): -0.5,
        }

        for (cause, effect), weight in causal_rules.items():
            if cause in nodes and effect in nodes:
                edges.append((cause, effect))
                edge_weights[(cause, effect)] = weight

        self.causal_graph = CausalGraph(
            nodes=nodes, edges=edges, edge_weights=edge_weights
        )

    def _estimate_causal_effects(self):
        """Estimate causal effects using various methods"""
        if self.data is None or self.causal_graph is None:
            return

        for cause, effect in self.causal_graph.edges:
            if cause in self.data.columns and effect in self.data.columns:
                # Simple linear regression for causal effect estimation
                try:
                    X = self.data[[cause]].values
                    y = self.data[effect].values

                    model = LinearRegression()
                    model.fit(X, y)

                    # Calculate effect size and confidence
                    ate = model.coef_[0]

                    # Simple confidence interval estimation
                    residuals = y - model.predict(X)
                    std_error = np.std(residuals) / np.sqrt(len(X))
                    ci_lower = ate - 1.96 * std_error
                    ci_upper = ate + 1.96 * std_error

                    # Simple p-value approximation
                    t_stat = ate / std_error
                    p_value = 2 * (
                        1
                        - abs(
                            0.5
                            * (
                                1
                                + np.sign(t_stat)
                                * (
                                    np.sqrt(2 / np.pi) * np.exp(-(t_stat**2) / 2)
                                    + t_stat
                                    * (
                                        1
                                        - 0.5
                                        * (
                                            1
                                            + np.sign(t_stat)
                                            * (1 - np.exp(-2 * t_stat**2 / np.pi))
                                        )
                                    )
                                )
                            )
                        )
                    )

                    self.causal_effects[f"{cause}->{effect}"] = CausalEffect(
                        treatment=cause,
                        outcome=effect,
                        ate=ate,
                        confidence_interval=(ci_lower, ci_upper),
                        p_value=min(p_value, 1.0),
                        method="linear_regression",
                        assumptions=["linearity", "no_unmeasured_confounders"],
                    )

                except Exception as e:
                    logger.warning(
                        f"Failed to estimate effect for {cause}->{effect}: {e}"
                    )

    def explain_anomaly(self, anomaly_state: Dict[str, float]) -> Dict[str, Any]:
        """Explain an anomaly using causal inference"""
        if self.causal_graph is None:
            return {"error": "Causal model not fitted"}

        # Find root causes
        root_causes = self._find_root_causes(anomaly_state)

        # Generate explanation
        explanation = {
            "anomaly_summary": self._summarize_anomaly(anomaly_state),
            "root_causes": root_causes,
            "causal_pathways": self._trace_causal_pathways(root_causes),
            "recommendations": self._generate_recommendations(root_causes),
            "confidence_scores": self._calculate_confidence_scores(root_causes),
        }

        return explanation

    def _find_root_causes(
        self, anomaly_state: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Find root causes of anomaly using causal graph"""
        root_causes = []

        # Identify anomalous parameters
        anomalous_params = []
        for param, value in anomaly_state.items():
            if param in self.parameters and abs(value) > 2.0:  # Z-score > 2
                anomalous_params.append(param)

        # Trace back causes using causal graph
        for param in anomalous_params:
            causes = self._find_upstream_causes(param, anomaly_state)
            if causes:
                root_causes.extend(causes)

        # Remove duplicates and sort by importance
        unique_causes = []
        seen = set()
        for cause in root_causes:
            cause_key = cause["parameter"]
            if cause_key not in seen:
                unique_causes.append(cause)
                seen.add(cause_key)

        return sorted(unique_causes, key=lambda x: x["importance"], reverse=True)[:5]

    def _find_upstream_causes(
        self, target_param: str, anomaly_state: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Find upstream causes for a target parameter"""
        causes = []

        if self.causal_graph is None:
            return causes

        # Find all nodes that can reach the target
        upstream_nodes = []
        for edge in self.causal_graph.edges:
            if edge[1] == target_param:
                upstream_nodes.append(edge[0])

        # Evaluate each potential cause
        for cause in upstream_nodes:
            if cause in anomaly_state:
                # Get causal effect
                effect_key = f"{cause}->{target_param}"
                if effect_key in self.causal_effects:
                    effect = self.causal_effects[effect_key]

                    # Calculate importance
                    importance = abs(effect.ate) * abs(anomaly_state[cause])

                    causes.append(
                        {
                            "parameter": cause,
                            "target": target_param,
                            "effect_size": effect.ate,
                            "importance": importance,
                            "confidence": 1.0 - effect.p_value,
                            "mechanism": f"{cause} affects {target_param} with effect size {effect.ate:.3f}",
                        }
                    )

        return causes

    def _trace_causal_pathways(
        self, root_causes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Trace causal pathways from root causes to failures"""
        pathways = []

        for cause in root_causes:
            pathway = {
                "root_cause": cause["parameter"],
                "pathway": [],
                "final_effects": [],
            }

            # Build pathway using BFS
            visited = set()
            queue = [(cause["parameter"], [cause["parameter"]])]

            while queue:
                current, path = queue.pop(0)

                if current in visited:
                    continue

                visited.add(current)

                # Find downstream effects
                for edge in self.causal_graph.edges or []:
                    if edge[0] == current:
                        next_node = edge[1]
                        new_path = path + [next_node]

                        if next_node in self.failure_indicators:
                            pathway["final_effects"].append(next_node)
                        else:
                            queue.append((next_node, new_path))

            pathway["pathway"] = pathway["pathway"][:5]  # Limit length
            pathways.append(pathway)

        return pathways

    def _generate_recommendations(self, root_causes: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on root causes"""
        recommendations = []

        for cause in root_causes:
            param = cause["parameter"]

            if param == "bearing_wear":
                recommendations.append("Schedule bearing inspection and replacement")
                recommendations.append("Check lubrication system and oil quality")
            elif param == "temperature":
                recommendations.append("Check cooling system and ventilation")
                recommendations.append("Reduce load or improve heat dissipation")
            elif param == "vibration_mm_s":
                recommendations.append("Inspect mechanical alignment and balance")
                recommendations.append("Check for loose components or mounting issues")
            elif param == "current_a":
                recommendations.append("Check electrical connections and insulation")
                recommendations.append("Verify voltage supply quality")
            elif param == "lubrication":
                recommendations.append("Replenish or replace lubricant")
                recommendations.append("Inspect lubrication delivery system")

        # Remove duplicates
        return list(set(recommendations))

    def _calculate_confidence_scores(
        self, root_causes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for root causes"""
        scores = {}

        for cause in root_causes:
            param = cause["parameter"]

            # Base confidence from causal effect
            base_confidence = cause["confidence"]

            # Adjust based on effect size and importance
            effect_confidence = min(1.0, abs(cause["effect_size"]) * 2)
            importance_confidence = min(1.0, cause["importance"] / 10)

            # Combined confidence
            combined_confidence = (
                base_confidence + effect_confidence + importance_confidence
            ) / 3

            scores[param] = combined_confidence

        return scores

    def _summarize_anomaly(self, anomaly_state: Dict[str, float]) -> str:
        """Generate summary of anomaly"""
        # Count anomalous parameters
        anomalous_count = sum(
            1
            for param, value in anomaly_state.items()
            if param in self.parameters and abs(value) > 2.0
        )

        # Find most anomalous parameter
        most_anomalous = max(
            (
                (param, abs(value))
                for param, value in anomaly_state.items()
                if param in self.parameters
            ),
            key=lambda x: x[1],
            default=("unknown", 0),
        )

        return f"Detected {anomalous_count} anomalous parameters, most severe: {most_anomalous[0]} (z-score: {most_anomalous[1]:.2f})"

    def get_causal_graph_visualization(self) -> Optional[nx.DiGraph]:
        """Get NetworkX graph for visualization"""
        if self.causal_graph is None:
            return None

        G = nx.DiGraph()

        # Add nodes
        G.add_nodes_from(self.causal_graph.nodes)

        # Add edges with weights
        for edge in self.causal_graph.edges:
            weight = self.causal_graph.edge_weights.get(edge, 1.0)
            G.add_edge(edge[0], edge[1], weight=weight)

        return G

    def export_causal_model(self, filepath: str):
        """Export causal model to file"""
        if self.causal_graph is None:
            logger.warning("No causal model to export")
            return

        export_data = {
            "nodes": self.causal_graph.nodes,
            "edges": self.causal_graph.edges,
            "edge_weights": self.causal_graph.edge_weights,
            "causal_effects": {
                key: {
                    "treatment": effect.treatment,
                    "outcome": effect.outcome,
                    "ate": effect.ate,
                    "confidence_interval": effect.confidence_interval,
                    "p_value": effect.p_value,
                    "method": effect.method,
                }
                for key, effect in self.causal_effects.items()
            },
        }

        import json

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Causal model exported to {filepath}")

    def get_explanation_summary(self) -> Dict[str, Any]:
        """Get summary of causal inference results"""
        if self.causal_graph is None:
            return {"status": "Model not fitted"}

        # Summary statistics
        total_effects = len(self.causal_effects)
        significant_effects = sum(
            1 for effect in self.causal_effects.values() if effect.p_value < 0.05
        )

        # Most influential parameters
        param_influence = {}
        for effect_key, effect in self.causal_effects.items():
            param = effect.treatment
            if param not in param_influence:
                param_influence[param] = []
            param_influence[param].append(abs(effect.ate))

        avg_influence = {
            param: np.mean(influences) for param, influences in param_influence.items()
        }

        most_influential = sorted(
            avg_influence.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "status": "Model fitted",
            "total_parameters": len(self.parameters),
            "total_causal_effects": total_effects,
            "significant_effects": significant_effects,
            "most_influential_parameters": most_influential,
            "graph_nodes": len(self.causal_graph.nodes),
            "graph_edges": len(self.causal_graph.edges),
        }
