#!/usr/bin/env python3
"""
Vehicle Trigger Module - Three-layer funnel trigger system
Implements Rule-based, Model-based, and Uncertainty-based triggers
"""

import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from config import Config


@dataclass
class CANSignal:
    """CAN bus vehicle signal"""
    timestamp_ms: int
    vehicle_speed_kmh: float
    longitudinal_accel_mps2: float
    lateral_accel_mps2: float
    brake_pedal_pos_pct: float
    steering_angle_deg: float
    steering_angle_velocity_deg_s: float  # 角速度
    autopilot_status: str  # ENGAGED, DISENGAGED, ERROR
    takeover_request: bool
    throttle_pos_pct: float = 0.0


@dataclass
class SensorData:
    """Multi-sensor data snapshot"""
    timestamp_ms: int
    camera_front_long: bytes  # Compressed image
    lidar_front_pointcloud: bytes  # XYZI point cloud
    radar_front_raw: bytes  # Radar point list
    gps_rtk: Dict[str, float]  # lat, lon, alt, heading
    imu_data: Dict[str, float]


@dataclass
class PerceptionOutput:
    """Perception model output"""
    timestamp_ms: int
    objects: List[Dict[str, Any]]
    max_confidence: float
    ood_score: float  # Out-of-distribution score
    trajectory_error: float  # Prediction deviation


@dataclass
class PlanningDecision:
    """Planning model output"""
    timestamp_ms: int
    current_lane: str
    target_speed: float
    obstacle_avoidance_plan: str
    reason: str
    control_command: Dict[str, float]


@dataclass
class ShadowModelDecision:
    """Shadow model decision"""
    timestamp_ms: int
    target_speed: float
    plan: str
    reason: str
    confidence: float


@dataclass
class TriggerMetadata:
    """Complete trigger event metadata"""
    event_id: str
    vehicle_id: str
    trigger_type: str
    trigger_time_utc: str
    location: Dict[str, Any]
    scenario_tags: List[str]
    trigger_details: Dict[str, Any]
    data_window: Dict[str, int]
    sensor_status: Dict[str, str]
    file_manifest: List[Dict[str, Any]]


class RuleBasedTrigger:
    """
    First Layer: Hard Rule-based Trigger
    Monitors CAN bus signals for emergency events
    """

    def __init__(self):
        self.speed_buffer = []  # For speed difference calculation
        self.max_buffer_size = 10  # Store 1s of data at 100Hz

    def check(self, can_signal: CANSignal) -> tuple[bool, Optional[str]]:
        """
        Check if rule-based trigger conditions are met

        Returns:
            (should_trigger, reason)
        """
        # Update speed buffer
        self._update_speed_buffer(can_signal.vehicle_speed_kmh)

        # Rule 1: Emergency brake (longitudinal acceleration < -4.0 m/s²)
        if (can_signal.longitudinal_accel_mps2 < Config.VehicleTrigger.RULE_LONGITUDINAL_ACCEL_THRESHOLD
                and can_signal.vehicle_speed_kmh > Config.VehicleTrigger.RULE_SPEED_THRESHOLD):
            return True, f"EMERGENCY_BRAKE: accel={can_signal.longitudinal_accel_mps2} m/s², speed={can_signal.vehicle_speed_kmh} km/h"

        # Rule 2: Sharp turn (lateral acceleration > 3.0 m/s²)
        if (can_signal.lateral_accel_mps2 > Config.VehicleTrigger.RULE_LATERAL_ACCEL_THRESHOLD
                and can_signal.vehicle_speed_kmh > Config.VehicleTrigger.RULE_SPEED_THRESHOLD):
            return True, f"SHARP_TURN: lateral_accel={can_signal.lateral_accel_mps2} m/s², speed={can_signal.vehicle_speed_kmh} km/h"

        # Rule 3: Speed delta > 10 km/h (1s) with brake > 20%
        speed_diff_1s = self._get_speed_diff_1s()
        if (speed_diff_1s > Config.VehicleTrigger.RULE_SPEED_DIFF_1S_THRESHOLD_1
                and can_signal.brake_pedal_pos_pct > Config.VehicleTrigger.RULE_BRAKE_THRESHOLD_1):
            return True, f"RAPID_DECEL_1: speed_diff={speed_diff_1s} km/h, brake={can_signal.brake_pedal_pos_pct}%"

        # Rule 4: Speed delta > 30 km/h (1s) with brake > 40%
        if (speed_diff_1s > Config.VehicleTrigger.RULE_SPEED_DIFF_1S_THRESHOLD_2
                and can_signal.brake_pedal_pos_pct > Config.VehicleTrigger.RULE_BRAKE_THRESHOLD_2):
            return True, f"RAPID_DECEL_2: speed_diff={speed_diff_1s} km/h, brake={can_signal.brake_pedal_pos_pct}%"

        # Rule 5: Steering angle velocity spike or brake > 80% (AEB precursory)
        if abs(can_signal.steering_angle_velocity_deg_s) > 500:  # deg/s
            return True, f"STEERING_SPIKE: angle_velocity={can_signal.steering_angle_velocity_deg_s} deg/s"

        if can_signal.brake_pedal_pos_pct > Config.VehicleTrigger.RULE_BRAKE_THRESHOLD_AEB:
            return True, f"AEB_PRECURSORY: brake={can_signal.brake_pedal_pos_pct}%"

        # Rule 6: Human takeover request
        if can_signal.takeover_request:
            return True, "HUMAN_TAKEOVER: driver intervention requested"

        return False, None

    def _update_speed_buffer(self, speed: float):
        """Update speed buffer for delta calculation"""
        self.speed_buffer.append(speed)
        if len(self.speed_buffer) > self.max_buffer_size:
            self.speed_buffer.pop(0)

    def _get_speed_diff_1s(self) -> float:
        """Calculate speed difference over 1 second"""
        if len(self.speed_buffer) < 2:
            return 0.0
        return self.speed_buffer[0] - self.speed_buffer[-1]


class ModelBasedTrigger:
    """
    Second Layer: Lightweight Perception Anomaly Detection
    Runs quantized model (INT8/FP16) for anomaly detection
    """

    def __init__(self):
        # Simulated model - in production this would load actual models
        self.detection_model = None  # YOLOv5s quantized
        self.ood_detector = None  # AutoEncoder for OOD
        self.trajectory_predictor = None  # LSTM/GRU

    def check(self, perception_output: PerceptionOutput) -> tuple[bool, Optional[str]]:
        """
        Check if model-based trigger conditions are met

        Returns:
            (should_trigger, reason)
        """
        triggers = []

        # Check 1: Low detection confidence
        if perception_output.max_confidence < Config.VehicleTrigger.MODEL_DETECTION_CONFIDENCE_THRESHOLD:
            triggers.append(f"LOW_CONFIDENCE: max_conf={perception_output.max_confidence:.3f}")

        # Check 2: Out-of-distribution detection
        if perception_output.ood_score > Config.VehicleTrigger.MODEL_OOD_RECONSTRUCTION_ERROR_THRESHOLD:
            triggers.append(f"OOD_DETECTED: ood_score={perception_output.ood_score:.3f}")

        # Check 3: Trajectory prediction deviation
        if perception_output.trajectory_error > Config.VehicleTrigger.MODEL_TRAJECTORY_ERROR_THRESHOLD:
            triggers.append(f"TRAJECTORY_ERROR: error={perception_output.trajectory_error:.2f}m")

        if triggers:
            return True, " | ".join(triggers)

        return False, None

    def simulate_detection(self, image: np.ndarray) -> PerceptionOutput:
        """
        Simulate perception model output (for testing)
        In production, this runs actual inference
        """
        # Simulated detection results
        confidence = np.random.uniform(0.2, 0.95)
        ood_score = np.random.uniform(0.1, 0.8)
        trajectory_error = np.random.uniform(0.1, 3.0)

        objects = []
        if confidence > 0.3:
            objects.append({
                "id": 101,
                "class": self._random_class(),
                "bbox_3d": [15.2, 0.5, 45.0],
                "confidence": confidence,
                "velocity": [0, 0, 0]
            })

        return PerceptionOutput(
            timestamp_ms=int(time.time() * 1000),
            objects=objects,
            max_confidence=confidence,
            ood_score=ood_score,
            trajectory_error=trajectory_error
        )

    def _random_class(self) -> str:
        """Return random object class"""
        classes = ["CAR", "TRUCK", "PEDESTRIAN", "CYCLIST",
                   "UNKNOWN_STATIC", "CONSTRUCTION_BARRIER", "DEBRIS"]
        return np.random.choice(classes)


class UncertaintyTrigger:
    """
    Third Layer: Shadow Mode & Uncertainty Estimation
    Compares production vs shadow model decisions
    """

    def __init__(self):
        self.shadow_decision_buffer = {}
        self.production_history = []

    def check(self,
              production_decision: PlanningDecision,
              shadow_decision: Optional[ShadowModelDecision] = None,
              uncertainty_entropy: float = 0.0) -> tuple[bool, Optional[str]]:
        """
        Check if uncertainty-based trigger conditions are met

        Returns:
            (should_trigger, reason)
        """
        triggers = []

        # Check 1: Shadow mode divergence
        if shadow_decision is not None:
            if self._has_decision_divergence(production_decision, shadow_decision):
                triggers.append(
                    f"SHADOW_MISMATCH: production={production_decision.target_speed}km/h,"
                    f" shadow={shadow_decision.target_speed}km/h,"
                    f" plan_prod={production_decision.obstacle_avoidance_plan},"
                    f" plan_shadow={shadow_decision.plan}"
                )

        # Check 2: High uncertainty entropy
        if uncertainty_entropy > Config.VehicleTrigger.UNCERTAINTY_ENTROPY_THRESHOLD:
            triggers.append(f"HIGH_UNCERTAINTY: entropy={uncertainty_entropy:.3f}")

        if triggers:
            return True, " | ".join(triggers)

        return False, None

    def _has_decision_divergence(self, prod: PlanningDecision, shadow: ShadowModelDecision) -> bool:
        """Check if production and shadow decisions diverge significantly"""
        # Speed divergence threshold
        speed_diff = abs(prod.target_speed - shadow.target_speed)
        if speed_diff > 15.0:  # 15 km/h difference
            return True

        # Plan divergence
        if prod.obstacle_avoidance_plan != shadow.plan:
            return True

        return False

    def calculate_entropy(self, predictions: np.ndarray) -> float:
        """
        Calculate Shannon entropy of predictions for uncertainty estimation
        Uses Monte Carlo Dropout ensemble

        Args:
            predictions: (n_samples, n_classes) probability array

        Returns:
            Entropy value (higher = more uncertain)
        """
        # Mean prediction across MC samples
        mean_pred = np.mean(predictions, axis=0)

        # Calculate entropy
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10))

        # Normalize by log of number of classes
        normalized_entropy = entropy / np.log(len(mean_pred))

        return normalized_entropy


class SensorBuffer:
    """
    Circular buffer for sensor data
    Stores pre-trigger and post-trigger data
    """

    def __init__(self, pre_trigger_seconds: int, post_trigger_seconds: int, sample_rate: int = 10):
        """
        Args:
            pre_trigger_seconds: Seconds before trigger to store
            post_trigger_seconds: Seconds after trigger to store
            sample_rate: Samples per second
        """
        self.pre_trigger_seconds = pre_trigger_seconds
        self.post_trigger_seconds = post_trigger_seconds
        self.sample_rate = sample_rate
        self.buffer_size = (pre_trigger_seconds + post_trigger_seconds) * sample_rate

        self.can_signals = []  # CAN signal buffer
        self.sensor_data = []  # Sensor data buffer
        self.trigger_index = pre_trigger_seconds * sample_rate  # Index where trigger occurs

    def add_data(self, can_signal: CANSignal, sensor_data: SensorData):
        """Add new data point to buffer"""
        self.can_signals.append(can_signal)
        self.sensor_data.append(sensor_data)

        # Maintain buffer size
        if len(self.can_signals) > self.buffer_size:
            self.can_signals.pop(0)
            self.sensor_data.pop(0)

    def get_trigger_data(self) -> tuple[List[CANSignal], List[SensorData]]:
        """Get data around trigger event"""
        if len(self.can_signals) < self.trigger_index + 1:
            # Not enough data, return what we have
            return self.can_signals[:], self.sensor_data[:]

        return (
            self.can_signals[max(0, len(self.can_signals) - self.buffer_size):],
            self.sensor_data[max(0, len(self.sensor_data) - self.buffer_size):]
        )


class DataPackageBuilder:
    """
    Builds data package for upload to cloud
    Includes metadata and sensor raw data
    """

    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id

    def build(self,
              can_signals: List[CANSignal],
              sensor_data_list: List[SensorData],
              perception_output: PerceptionOutput,
              planning_decision: PlanningDecision,
              shadow_decision: Optional[ShadowModelDecision],
              trigger_type: str,
              trigger_reason: str,
              uncertainty_entropy: float = 0.0) -> Dict[str, Any]:
        """
        Build complete trigger data package

        Returns:
            Dictionary with metadata and raw data references
        """
        # Generate event ID
        event_id = f"evt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Extract location from GPS data
        location = self._extract_location(sensor_data_list)

        # Generate scenario tags
        scenario_tags = self._generate_scenario_tags(can_signals, perception_output, planning_decision)

        # Build file manifest
        file_manifest = self._build_file_manifest(sensor_data_list, can_signals, planning_decision)

        # Build metadata
        metadata = {
            "event_id": event_id,
            "vehicle_id": self.vehicle_id,
            "trigger_type": trigger_type,
            "trigger_time_utc": datetime.utcnow().isoformat() + "Z",
            "location": location,
            "scenario_tags": scenario_tags,
            "trigger_details": {
                "rule_hit": trigger_type,
                "production_model_decision": f"{planning_decision.obstacle_avoidance_plan} (speed: {planning_decision.target_speed}km/h)",
                "shadow_model_decision": f"{shadow_decision.plan} (speed: {shadow_decision.target_speed}km/h)" if shadow_decision else "N/A",
                "perception_max_confidence": perception_output.max_confidence,
                "uncertainty_entropy": uncertainty_entropy,
            },
            "data_window": {
                "pre_trigger_seconds": Config.VehicleTrigger.PRE_TRIGGER_SECONDS,
                "post_trigger_seconds": Config.VehicleTrigger.POST_TRIGGER_SECONDS,
                "total_duration_seconds": Config.VehicleTrigger.TOTAL_DURATION_SECONDS,
            },
            "sensor_status": self._get_sensor_status(sensor_data_list),
            "file_manifest": file_manifest,
        }

        return {
            "metadata": metadata,
            "can_signals": can_signals,
            "sensor_data": sensor_data_list,
            "perception_output": perception_output,
            "planning_decision": planning_decision,
            "shadow_decision": shadow_decision,
        }

    def _extract_location(self, sensor_data_list: List[SensorData]) -> Dict[str, Any]:
        """Extract location from sensor data"""
        if not sensor_data_list:
            return {
                "latitude": 0.0,
                "longitude": 0.0,
                "altitude": 0.0,
                "heading": 0.0,
                "road_name": "UNKNOWN",
                "lane_id": "UNKNOWN"
            }

        # Use middle of data window
        mid_index = len(sensor_data_list) // 2
        gps = sensor_data_list[mid_index].gps_rtk

        return {
            "latitude": gps.get("lat", 0.0),
            "longitude": gps.get("lon", 0.0),
            "altitude": gps.get("alt", 0.0),
            "heading": gps.get("heading", 0.0),
            "road_name": self._get_road_name(gps),
            "lane_id": self._get_lane_id(gps),
        }

    def _get_road_name(self, gps: Dict[str, float]) -> str:
        """Get road name from GPS (simplified)"""
        # In production, query HD map
        return "G4_Jinggangao_Expressway"

    def _get_lane_id(self, gps: Dict[str, float]) -> str:
        """Get lane ID from GPS (simplified)"""
        # In production, query HD map
        return "LN_G4_North_03"

    def _generate_scenario_tags(self,
                                 can_signals: List[CANSignal],
                                 perception: PerceptionOutput,
                                 planning: PlanningDecision) -> List[str]:
        """Generate scenario tags from data"""
        tags = []

        # Speed range
        avg_speed = np.mean([s.vehicle_speed_kmh for s in can_signals])
        if avg_speed > 80:
            tags.append("highway")
        elif avg_speed > 40:
            tags.append("urban_road")
        else:
            tags.append("low_speed")

        # Trigger conditions
        if planning.obstacle_avoidance_plan != "NONE":
            tags.append("obstacle_avoidance")

        if perception.ood_score > Config.VehicleTrigger.MODEL_OOD_RECONSTRUCTION_ERROR_THRESHOLD:
            tags.append("ood_scenario")

        # Weather (simplified - would use actual sensor data)
        tags.append("clear_weather")

        return tags

    def _build_file_manifest(self,
                             sensor_data_list: List[SensorData],
                             can_signals: List[CANSignal],
                             planning: PlanningDecision) -> List[Dict[str, Any]]:
        """Build list of files to upload"""
        manifest = []

        # Camera data
        if sensor_data_list:
            manifest.append({
                "file_name": "cam_front_long_00.bag",
                "type": "video_raw",
                "format": "h264",
                "size_bytes": len(sensor_data_list[0].camera_front_long),
                "s3_path": self._build_s3_path("cam_front_long_00.bag"),
            })

        # LiDAR data
        if sensor_data_list:
            manifest.append({
                "file_name": "lidar_front_pointcloud.bin",
                "type": "pointcloud_raw",
                "format": "pcap/bin",
                "size_bytes": len(sensor_data_list[0].lidar_front_pointcloud),
                "s3_path": self._build_s3_path("lidar_front_pointcloud.bin"),
            })

        # CAN bus signals
        can_csv = self._can_to_csv(can_signals)
        manifest.append({
            "file_name": "can_bus_signals.csv",
            "type": "vehicle_state",
            "format": "csv",
            "size_bytes": len(can_csv.encode()),
            "s3_path": self._build_s3_path("can_bus_signals.csv"),
        })

        # Planning log
        planning_json = json.dumps(asdict(planning), default=str)
        manifest.append({
            "file_name": "planning_log.json",
            "type": "system_log",
            "format": "json",
            "size_bytes": len(planning_json.encode()),
            "s3_path": self._build_s3_path("planning_log.json"),
        })

        return manifest

    def _can_to_csv(self, can_signals: List[CANSignal]) -> str:
        """Convert CAN signals to CSV format"""
        lines = ["timestamp_ms,vehicle_speed_kmh,longitudinal_accel_mps2,lateral_accel_mps2,"
                 "brake_pedal_pos_pct,steering_angle_deg,autopilot_status,takeover_request"]

        for sig in can_signals:
            lines.append(
                f"{sig.timestamp_ms},{sig.vehicle_speed_kmh:.1f},"
                f"{sig.longitudinal_accel_mps2:.2f},{sig.lateral_accel_mps2:.2f},"
                f"{sig.brake_pedal_pos_pct:.1f},{sig.steering_angle_deg:.1f},"
                f"{sig.autopilot_status},{sig.takeover_request}"
            )

        return "\n".join(lines)

    def _build_s3_path(self, filename: str) -> str:
        """Build S3 path for file"""
        date = datetime.utcnow()
        return (
            f"s3://{Config.Storage.S3_BUCKET_RAW}/"
            f"{date.strftime('%Y/%m/%d')}/"
            f"{self.vehicle_id}/"
            f"evt_{date.strftime('%Y%m%d_%H%M%S')}/"
            f"{filename}"
        )

    def _get_sensor_status(self, sensor_data_list: List[SensorData]) -> Dict[str, str]:
        """Get sensor status"""
        if not sensor_data_list:
            return {
                "lidar_front": "ERROR",
                "camera_front_long": "ERROR",
                "radar_front": "ERROR",
                "gps_rtk": "ERROR",
                "imu": "ERROR"
            }

        # Simplified - would check actual sensor health
        return {
            "lidar_front": "OK",
            "camera_front_long": "OK",
            "radar_front": "OK",
            "gps_rtk": "FIXED" if sensor_data_list[0].gps_rtk.get("fix", 0) > 0 else "FLOAT",
            "imu": "OK"
        }


class VehicleTriggerManager:
    """
    Main vehicle trigger manager
    Coordinates three-layer trigger funnel
    """

    def __init__(self, vehicle_id: str):
        self.vehicle_id = vehicle_id
        self.rule_trigger = RuleBasedTrigger()
        self.model_trigger = ModelBasedTrigger()
        self.uncertainty_trigger = UncertaintyTrigger()
        self.sensor_buffer = SensorBuffer(
            pre_trigger_seconds=Config.VehicleTrigger.PRE_TRIGGER_SECONDS,
            post_trigger_seconds=Config.VehicleTrigger.POST_TRIGGER_SECONDS
        )
        self.package_builder = DataPackageBuilder(vehicle_id)

    def process_frame(self,
                      can_signal: CANSignal,
                      sensor_data: SensorData,
                      perception_output: PerceptionOutput,
                      planning_decision: PlanningDecision,
                      shadow_decision: Optional[ShadowModelDecision] = None,
                      uncertainty_entropy: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Process a single frame through three-layer trigger funnel

        Returns:
            Data package if trigger occurs, None otherwise
        """
        # Add data to buffer
        self.sensor_buffer.add_data(can_signal, sensor_data)

        # Layer 1: Rule-based trigger
        should_trigger, rule_reason = self.rule_trigger.check(can_signal)
        trigger_type = "RULE_BASED"

        # Layer 2: Model-based trigger (if no rule trigger)
        if not should_trigger:
            should_trigger, model_reason = self.model_trigger.check(perception_output)
            if should_trigger:
                trigger_type = "MODEL_BASED"
                rule_reason = model_reason

        # Layer 3: Uncertainty trigger (if no previous trigger)
        if not should_trigger:
            should_trigger, uncertainty_reason = self.uncertainty_trigger.check(
                planning_decision, shadow_decision, uncertainty_entropy
            )
            if should_trigger:
                trigger_type = "UNCERTAINTY_BASED"
                rule_reason = uncertainty_reason

        # Build data package if triggered
        if should_trigger:
            can_signals, sensor_data_list = self.sensor_buffer.get_trigger_data()
            return self.package_builder.build(
                can_signals=can_signals,
                sensor_data_list=sensor_data_list,
                perception_output=perception_output,
                planning_decision=planning_decision,
                shadow_decision=shadow_decision,
                trigger_type=trigger_type,
                trigger_reason=rule_reason,
                uncertainty_entropy=uncertainty_entropy
            )

        return None


if __name__ == '__main__':
    # Test vehicle trigger module
    print("Vehicle Trigger Module Test")
    print("=" * 50)

    manager = VehicleTriggerManager("VIN_AD_TEST_007")

    # Simulate data stream
    for i in range(100):
        # Generate random CAN signal
        can_signal = CANSignal(
            timestamp_ms=int(time.time() * 1000),
            vehicle_speed_kmh=np.random.uniform(60, 100),
            longitudinal_accel_mps2=np.random.uniform(-5, 2),
            lateral_accel_mps2=np.random.uniform(-3, 3),
            brake_pedal_pos_pct=np.random.uniform(0, 50),
            steering_angle_deg=np.random.uniform(-30, 30),
            steering_angle_velocity_deg_s=np.random.uniform(-100, 100),
            autopilot_status="ENGAGED",
            takeover_request=False,
        )

        # Generate sensor data
        sensor_data = SensorData(
            timestamp_ms=can_signal.timestamp_ms,
            camera_front_long=b"mock_video_data",
            lidar_front_pointcloud=b"mock_pointcloud_data",
            radar_front_raw=b"mock_radar_data",
            gps_rtk={"lat": 39.9042, "lon": 116.4074, "alt": 52.5, "heading": 85.5, "fix": 1},
            imu_data={"ax": 0.1, "ay": 0.2, "az": 9.8, "wx": 0.01, "wy": 0.02, "wz": 0.01}
        )

        # Generate perception output
        perception = PerceptionOutput(
            timestamp_ms=can_signal.timestamp_ms,
            objects=[],
            max_confidence=np.random.uniform(0.2, 0.95),
            ood_score=np.random.uniform(0.1, 0.8),
            trajectory_error=np.random.uniform(0.1, 3.0)
        )

        # Generate planning decision
        planning = PlanningDecision(
            timestamp_ms=can_signal.timestamp_ms,
            current_lane="LANE_03",
            target_speed=80.0,
            obstacle_avoidance_plan="NONE",
            reason="No obstacle detected",
            control_command={"throttle": 0.2, "brake": 0.0, "steering": 0.0}
        )

        # Process frame
        result = manager.process_frame(can_signal, sensor_data, perception, planning)

        if result:
            print(f"\nTrigger Event: {result['metadata']['event_id']}")
            print(f"Type: {result['metadata']['trigger_type']}")
            print(f"Reason: {result['metadata']['trigger_details']['rule_hit']}")
            print(f"Confidence: {result['metadata']['trigger_details']['perception_max_confidence']:.2f}")
            print(f"Files: {len(result['metadata']['file_manifest'])}")

    print("\nTest completed!")
