#!/usr/bin/env python3
"""
Autopilot Data Loop System - Configuration
Configuration parameters for the vehicle data closed-loop pipeline
"""

import os
from typing import Dict, Any

class Config:
    """System-wide configuration"""

    # ==================== Vehicle Trigger Module ====================
    class VehicleTrigger:
        """Vehicle-side trigger configuration"""

        # First Layer: Rule-based thresholds
        RULE_LONGITUDINAL_ACCEL_THRESHOLD = -4.0  # m/s², emergency brake
        RULE_LATERAL_ACCEL_THRESHOLD = 3.0  # m/s², sharp turn
        RULE_SPEED_THRESHOLD = 10.0  # km/h, minimum speed for trigger
        RULE_SPEED_DIFF_1S_THRESHOLD_1 = 10.0  # km/h, with 20% brake
        RULE_SPEED_DIFF_1S_THRESHOLD_2 = 30.0  # km/h, with 40% brake
        RULE_BRAKE_THRESHOLD_1 = 20.0  # %
        RULE_BRAKE_THRESHOLD_2 = 40.0  # %
        RULE_BRAKE_THRESHOLD_AEB = 80.0  # %, AEB precursory

        # Second Layer: Model-based thresholds
        MODEL_DETECTION_CONFIDENCE_THRESHOLD = 0.4  # Low confidence trigger
        MODEL_OOD_RECONSTRUCTION_ERROR_THRESHOLD = 0.5  # Out-of-distribution
        MODEL_TRAJECTORY_ERROR_THRESHOLD = 2.0  # meters, prediction deviation

        # Third Layer: Uncertainty thresholds
        UNCERTAINTY_ENTROPY_THRESHOLD = 0.8  # High uncertainty trigger
        SHADOW_MODE_MISMATCH_TIMEOUT = 5.0  # seconds to wait for shadow decision

        # Data window configuration
        PRE_TRIGGER_SECONDS = 10
        POST_TRIGGER_SECONDS = 20
        TOTAL_DURATION_SECONDS = 30

    # ==================== Data Transmission ====================
    class Transmission:
        """Data transmission configuration"""

        # Size thresholds for protocol selection
        SMALL_PACKET_SIZE_KB = 100  # KB, use MQTT
        LARGE_PACKET_SIZE_MB = 500  # MB, use HTTP/2 for below, QUIC for above

        # MQTT configuration
        MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt.baidubce.com")
        MQTT_PORT = 8883  # TLS port
        MQTT_QOS = 1
        MQTT_TOPIC_METADATA = "vehicle/metadata"
        MQTT_TOPIC_STATUS = "vehicle/status"

        # gRPC/HTTP2 configuration
        HTTP2_ENDPOINT = os.getenv("HTTP2_ENDPOINT", "https://gateway.autopilot.com/api/v2/upload")
        UPLOAD_CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks

        # QUIC configuration
        QUIC_ENDPOINT = os.getenv("QUIC_ENDPOINT", "quic://gateway.autopilot.com:4433")

    # ==================== Kafka Configuration ====================
    class Kafka:
        """Apache Kafka configuration - Real producer->topic->consumer flow"""

        BOOTSTRAP_SERVERS = os.getenv("KAFKA_BROKERS", "localhost:9092")
        CLIENT_ID = "autopilot_data_loop"

        # Topic definitions
        TOPIC_METADATA = "autopilot.metadata"
        TOPIC_SENSOR_RAW = "autopilot.sensor.raw"
        TOPIC_STATUS = "autopilot.vehicle.status"
        TOPIC_ALERT = "autopilot.alert"

        # Topics for data processing
        TOPIC_PROCESSED = "autopilot.processed"
        TOPIC_ANNOTATION_REQUEST = "autopilot.annotation.request"
        TOPIC_ANNOTATION_RESULT = "autopilot.annotation.result"

        # Producer configuration
        PRODUCER_CONFIG = {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "client.id": CLIENT_ID + "_producer",
            "acks": "all",  # Wait for all replicas
            "retries": 3,
            "compression.type": "snappy",
            "linger.ms": 5,
            "batch.size": 32768,
            "enable.idempotence": True,
        }

        # Consumer configuration
        CONSUMER_CONFIG = {
            "bootstrap.servers": BOOTSTRAP_SERVERS,
            "client.id": CLIENT_ID + "_consumer",
            "group.id": "autopilot_consumer_group",
            "auto.offset.reset": "latest",
            "enable.auto.commit": True,
            "max.poll.records": 100,
            "session.timeout.ms": 30000,
        }

        # Topic partitioning
        METADATA_PARTITIONS = 3
        SENSOR_RAW_PARTITIONS = 6
        REPLICATION_FACTOR = 2

    # ==================== Flink Configuration ====================
    class Flink:
        """Apache Flink stream processing configuration"""

        CHECKPOINTING_INTERVAL_MS = 60000  # 1 minute
        CHECKPOINT_TIMEOUT_MS = 300000  # 5 minutes
        STATE_BACKEND = "rocksdb"

        # Window configuration
        WINDOW_SIZE_SECONDS = 60
        WINDOW_SLIDE_SECONDS = 10

        # Alert thresholds
        AEB_TRIGGER_THRESHOLD_COUNT = 3  # Count within window to trigger alert
        HIGH_RISK_SCENARIO_THRESHOLD = 5

    # ==================== Spark Configuration ====================
    class Spark:
        """Apache Spark batch processing configuration"""

        APP_NAME = "AutopilotDataETL"
        MASTER = "yarn"  # or "local[*]" for local testing

        # Data cleaning thresholds
        TIME_SYNC_TOLERANCE_MS = 50  # Tolerance for sensor time alignment
        SENSOR_HEALTH_CHECK_INTERVAL_HOURS = 6

        # ETL batch schedule
        ETL_SCHEDULE_HOUR = 2  # Run at 2 AM daily

    # ==================== Storage Configuration ====================
    class Storage:
        """Tiered storage configuration"""

        # S3/OSS configuration
        S3_ENDPOINT = os.getenv("S3_ENDPOINT", "s3.amazonaws.com")
        S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
        S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
        S3_BUCKET_RAW = "ad-data-lake-raw"
        S3_BUCKET_PROCESSED = "ad-data-lake-processed"
        S3_BUCKET_MODELS = "ad-models"

        # Path templates
        RAW_DATA_PATH_TEMPLATE = "bucket/{date:%Y/%m/%d}/{vehicle_id}/{event_id}/"
        PROCESSED_DATA_PATH_TEMPLATE = "processed/{date:%Y/%m/%d}/{scenario_type}/"

        # IoTDB (time-series) configuration
        IOTDB_HOST = os.getenv("IOTDB_HOST", "localhost")
        IOTDB_PORT = 6667
        IOTDB_USERNAME = "root"
        IOTDB_PASSWORD = "root"
        DATABASE_CAN = "autopilot_can"
        DATABASE_GPS = "autopilot_gps"

        # PostgreSQL (metadata/index) configuration
        PG_HOST = os.getenv("PG_HOST", "localhost")
        PG_PORT = 5432
        PG_DATABASE = "autopilot_metadata"
        PG_USERNAME = "autopilot"
        PG_PASSWORD = os.getenv("PG_PASSWORD", "")

    # ==================== Annotation Configuration ====================
    class Annotation:
        """Human-in-the-loop annotation configuration"""

        # Confidence thresholds for routing
        CONFIDENCE_HIGH = 0.9  # Direct入库，免检
        CONFIDENCE_LOW = 0.4  # 全量复核

        # Pre-annotation model
        FOUNDATION_MODEL_NAME = "foundation-vision-large-v3"

        # Annotation content
        ANNOTATION_TYPES = [
            "bbox_2d",      # 2D bounding box
            "bbox_3d",      # 3D bounding box
            "segmentation", # Semantic segmentation
            "tracking",     # Multi-object tracking
            "trajectory",   # Future trajectory
            "expert_demonstration",  # Expert replay for control
        ]

    # ==================== Data Augmentation Configuration ====================
    class Augmentation:
        """Data augmentation configuration"""

        # Traditional augmentation
        TRADITIONAL_METHODS = [
            "geometric_transform",
            "color_jitter",
            "noise_injection",
            "weather_simulation",
        ]

        # AIGC augmentation
        NERF_ITERATIONS = 10000
        DIFFUSION_STEPS = 50

        # Simulation generation
        SIMULATION_VARIANTS_PER_SCENE = 1000

    # ==================== Training Configuration ====================
    class Training:
        """Model training configuration"""

        # Distributed training
        DDP_BACKEND = "nccl"
        NUM_GPUS = 8
        BATCH_SIZE_PER_GPU = 32

        # Training parameters
        EPOCHS = 50
        LEARNING_RATE = 1e-4
        WEIGHT_DECAY = 1e-5
        GRADIENT_CLIP = 1.0

        # Checkpointing
        SAVE_INTERVAL_STEPS = 1000
        MAX_CHECKPOINTS = 5

    # ==================== Simulation Configuration ====================
    class Simulation:
        """Closed-loop simulation configuration"""

        SIMULATOR = "carla"  # or "vtd", "custom"
        SIMULATION_HOST = "localhost"
        SIMULATION_PORT = 2000

        # Regression test
        REGRESSION_TEST_PASSES = 100  # Run each failure case 100 times

        # Stress test
        STRESS_TEST_SCENARIOS = 1000000  # 1M random scenarios
        PASS_RATE_THRESHOLD = 0.98  # 98% pass rate required

    # ==================== OTA Configuration ====================
    class OTA:
        """OTA deployment configuration"""

        OTA_SERVER = "ota.autopilot.com"
        GRAYSCALE_PERCENTAGE_START = 5  # Start with 5% of fleet
        GRAYSCALE_INCREMENT = 10  # Add 10% each phase
        GRAYSCALE_PHASE_DURATION_HOURS = 24  # Each phase lasts 24 hours

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert all configuration to dictionary"""
        config = {}
        for name in dir(cls):
            if not name.startswith('_') and name.isupper():
                attr = getattr(cls, name)
                if hasattr(attr, '__dict__'):
                    config[name] = {k: v for k, v in attr.__dict__.items()
                                   if not k.startswith('_')}
        return config


if __name__ == '__main__':
    # Print configuration for debugging
    import json
    print(json.dumps(Config.to_dict(), indent=2, default=str))
