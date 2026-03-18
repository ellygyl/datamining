#!/usr/bin/env python3
"""
Autopilot Data Loop System - Main Orchestrator
Coordinates the complete "车端异常触发 -> 云端挖掘 -> 模型训练" pipeline
"""

import json
import logging
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import all modules
from config import Config
from vehicle_trigger import (
    VehicleTriggerManager,
    CANSignal,
    SensorData,
    PerceptionOutput,
    PlanningDecision,
    ShadowModelDecision
)
from cloud_edge import CloudEdgeOrchestrator
from stream_batch import StreamBatchOrchestrator
from data_mining import DataMiningOrchestrator
from training_validation import (
    TrainingValidationOrchestrator,
    TrainingConfig,
    ModelType
)


class AutopilotDataLoopOrchestrator:
    """
    Main orchestrator for the Autopilot Data Closed-Loop System

    Pipeline:
    1. Vehicle Trigger Module (vehicle_trigger.py)
       - Rule-based trigger
       - Model-based trigger
       - Uncertainty-based trigger
       - Data package builder

    2. Cloud Edge Module (cloud_edge.py)
       - Transmission router (MQTT/gRPC/QUIC)
       - Cloud gateway with certificate validation
       - Kafka producer (producer->topic->consumer)
       - Storage manager (S3/OSS, IoTDB, PostgreSQL)

    3. Stream Batch Processing Module (stream_batch.py)
       - Kafka consumer group
       - Flink stream processor (KeyBy, Window, Process)
       - Spark ETL processor
       - Real-time alerts

    4. Data Mining Module (data_mining.py)
       - Scenario mining (rule, cluster, hard)
       - Human-in-the-loop annotation
       - Data augmentation (traditional + AIGC)
       - Dataset building

    5. Training Validation Module (training_validation.py)
       - Model training (DDP/FSDP)
       - Model evaluation
       - Closed-loop simulation
       - OTA grayscale deployment
    """

    def __init__(self, vehicle_id: str = "VIN_DEFAULT"):
        """
        Initialize the orchestrator

        Args:
            vehicle_id: Vehicle identifier for this instance
        """
        self.vehicle_id = vehicle_id

        # Initialize all module orchestrators
        self.vehicle_trigger_manager = VehicleTriggerManager(vehicle_id)
        self.cloud_edge_orchestrator = CloudEdgeOrchestrator()
        self.stream_batch_orchestrator = StreamBatchOrchestrator()
        self.data_mining_orchestrator = DataMiningOrchestrator()
        self.training_orchestrator = TrainingValidationOrchestrator()

        # Pipeline state
        self.events_buffer: List[Dict[str, Any]] = []
        self.processed_events: List[Dict[str, Any]] = []
        self.pipeline_status = "initialized"

        logging.info(f"AutopilotDataLoopOrchestrator initialized for {vehicle_id}")

    def initialize(self):
        """Initialize all components"""
        logging.info("Initializing Autopilot Data Loop System...")

        # Initialize cloud edge
        self.cloud_edge_orchestrator.initialize()

        # Start stream processing
        self.stream_batch_orchestrator.start_stream_processing()

        self.pipeline_status = "running"
        logging.info("System initialization complete")

    def process_vehicle_frame(self,
                             can_signal: CANSignal,
                             sensor_data: SensorData,
                             perception_output: PerceptionOutput,
                             planning_decision: PlanningDecision,
                             shadow_decision: Optional[ShadowModelDecision] = None,
                             uncertainty_entropy: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Process a single vehicle frame through the complete pipeline

        This is the main entry point for real-time vehicle data processing.

        Args:
            can_signal: CAN bus signals
            sensor_data: Multi-sensor data
            perception_output: Perception model output
            planning_decision: Planning model decision
            shadow_decision: Shadow model decision (optional)
            uncertainty_entropy: Uncertainty entropy value

        Returns:
            Event metadata if trigger occurred, None otherwise
        """
        # Step 1: Check triggers (Three-layer funnel)
        data_package = self.vehicle_trigger_manager.process_frame(
            can_signal=can_signal,
            sensor_data=sensor_data,
            perception_output=perception_output,
            planning_decision=planning_decision,
            shadow_decision=shadow_decision,
            uncertainty_entropy=uncertainty_entropy
        )

        if data_package is None:
            return None

        # Step 2: Upload to cloud (Kafka + Storage)
        upload_results = self.cloud_edge_orchestrator.upload_event(data_package)

        # Step 3: Stream processing handles the uploaded data asynchronously
        # (handled by Kafka consumer in stream_batch module)

        # Store event for batch processing
        self.events_buffer.append(data_package["metadata"])

        logging.info(f"Event processed: {data_package['metadata']['event_id']}, "
                    f"upload_success={all(upload_results.values())}")

        return data_package["metadata"]

    def run_batch_pipeline(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Run the batch processing pipeline

        This includes:
        1. ETL processing
        2. Data mining
        3. Annotation
        4. Dataset building

        Args:
            days_back: Number of days to process

        Returns:
            Pipeline results
        """
        logging.info(f"Running batch pipeline for {days_back} days...")

        results = {
            "etl": {},
            "mining": {},
            "training": {},
        }

        # Step 1: Run ETL
        etl_results = self.stream_batch_orchestrator.run_batch_etl(days_back)
        results["etl"] = etl_results

        # Step 2: Run data mining on accumulated events
        if self.events_buffer:
            mining_results = self.data_mining_orchestrator.process_pipeline(self.events_buffer)
            results["mining"] = mining_results

        # Step 3: Run training if enough data
        if results["mining"].get("annotated_events", 0) >= 10:
            training_config = TrainingConfig(
                model_type=ModelType.PERCEPTION,
                model_name=f"perception_v{int(time.time())}",
                epochs=Config.Training.EPOCHS,
            )

            train_data = self.events_buffer[:int(len(self.events_buffer) * 0.7)]
            val_data = self.events_buffer[int(len(self.events_buffer) * 0.7):int(len(self.events_buffer) * 0.85)]
            test_data = self.events_buffer[int(len(self.events_buffer) * 0.85):]

            training_results = self.training_orchestrator.run_training_pipeline(
                train_data, val_data, test_data, training_config
            )
            results["training"] = training_results

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "vehicle_id": self.vehicle_id,
            "pipeline_status": self.pipeline_status,
            "events_buffered": len(self.events_buffer),
            "events_processed": len(self.processed_events),
            "stream_metrics": self.stream_batch_orchestrator.flink_processor.get_metrics(),
            "dashboard": self.stream_batch_orchestrator.get_dashboard_data(),
        }

    def shutdown(self):
        """Shutdown all components"""
        logging.info("Shutting down Autopilot Data Loop System...")

        self.cloud_edge_orchestrator.shutdown()
        self.stream_batch_orchestrator.shutdown()

        self.pipeline_status = "shutdown"
        logging.info("System shutdown complete")


def run_demo():
    """Run a demonstration of the complete pipeline"""
    print("=" * 70)
    print("Autopilot Data Closed-Loop System - Demo")
    print("=" * 70)

    # Initialize orchestrator
    orchestrator = AutopilotDataLoopOrchestrator(vehicle_id="VIN_DEMO_001")
    orchestrator.initialize()

    # Simulate vehicle data stream
    print("\n[1] Simulating vehicle data stream...")
    import numpy as np

    triggered_events = []
    for i in range(100):
        # Generate simulated data
        can_signal = CANSignal(
            timestamp_ms=int(time.time() * 1000) + i * 100,
            vehicle_speed_kmh=np.random.uniform(60, 100),
            longitudinal_accel_mps2=np.random.uniform(-5, 2),
            lateral_accel_mps2=np.random.uniform(-3, 3),
            brake_pedal_pos_pct=np.random.uniform(0, 50),
            steering_angle_deg=np.random.uniform(-30, 30),
            steering_angle_velocity_deg_s=np.random.uniform(-100, 100),
            autopilot_status="ENGAGED",
            takeover_request=False,
        )

        sensor_data = SensorData(
            timestamp_ms=can_signal.timestamp_ms,
            camera_front_long=b"mock_camera_data",
            lidar_front_pointcloud=b"mock_lidar_data",
            radar_front_raw=b"mock_radar_data",
            gps_rtk={"lat": 39.9042, "lon": 116.4074, "alt": 52.5, "heading": 85.5, "fix": 1},
            imu_data={"ax": 0.1, "ay": 0.2, "az": 9.8}
        )

        perception_output = PerceptionOutput(
            timestamp_ms=can_signal.timestamp_ms,
            objects=[],
            max_confidence=np.random.uniform(0.2, 0.95),
            ood_score=np.random.uniform(0.1, 0.8),
            trajectory_error=np.random.uniform(0.1, 3.0)
        )

        planning_decision = PlanningDecision(
            timestamp_ms=can_signal.timestamp_ms,
            current_lane="LANE_03",
            target_speed=80.0,
            obstacle_avoidance_plan="NONE",
            reason="No obstacle detected",
            control_command={"throttle": 0.2, "brake": 0.0, "steering": 0.0}
        )

        shadow_decision = ShadowModelDecision(
            timestamp_ms=can_signal.timestamp_ms,
            target_speed=75.0,
            plan="NONE",
            reason="Shadow model agrees",
            confidence=0.8
        ) if np.random.random() > 0.7 else None

        # Process frame
        event_metadata = orchestrator.process_vehicle_frame(
            can_signal=can_signal,
            sensor_data=sensor_data,
            perception_output=perception_output,
            planning_decision=planning_decision,
            shadow_decision=shadow_decision,
            uncertainty_entropy=np.random.uniform(0.1, 0.9)
        )

        if event_metadata:
            triggered_events.append(event_metadata)
            print(f"  Trigger #{len(triggered_events)}: {event_metadata['event_id']} "
                  f"(type: {event_metadata['trigger_type']})")

    print(f"\n  Total triggered events: {len(triggered_events)}")

    # Show system status
    print("\n[2] System Status:")
    status = orchestrator.get_system_status()
    print(f"  Pipeline Status: {status['pipeline_status']}")
    print(f"  Events Buffered: {status['events_buffered']}")
    print(f"  Events Processed: {status['stream_metrics']['events_processed']}")
    print(f"  Alerts Generated: {status['stream_metrics']['alerts_generated']}")

    # Run batch pipeline
    print("\n[3] Running Batch Pipeline...")
    batch_results = orchestrator.run_batch_pipeline(days_back=1)
    print(f"  ETL Status: {batch_results['etl'].get('status', 'N/A')}")
    print(f"  Mined Scenarios: {batch_results['mining'].get('mined_scenarios', 0)}")
    print(f"  Annotated Events: {batch_results['mining'].get('annotated_events', 0)}")
    print(f"  Augmented Events: {batch_results['mining'].get('augmented_events', 0)}")

    if batch_results.get('training'):
        print(f"  Training Status: {batch_results['training']['training']['status']}")
        print(f"  Final mAP: {batch_results['training']['evaluation'].get('mAP', 0):.3f}")
        print(f"  OTA Status: {batch_results['training']['ota'].get('status', 'N/A')}")

    # Shutdown
    print("\n[4] Shutting down...")
    orchestrator.shutdown()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)

    return batch_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Autopilot Data Closed-Loop System")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--vehicle-id", default="VIN_DEFAULT", help="Vehicle ID")
    parser.add_argument("--config", help="Path to config file")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.demo:
        return run_demo()

    # Normal operation
    orchestrator = AutopilotDataLoopOrchestrator(vehicle_id=args.vehicle_id)
    orchestrator.initialize()

    try:
        while True:
            status = orchestrator.get_system_status()
            print(f"Status: {status['pipeline_status']}, "
                  f"Buffered: {status['events_buffered']}")
            time.sleep(10)
    except KeyboardInterrupt:
        orchestrator.shutdown()


if __name__ == '__main__':
    main()
