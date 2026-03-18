#!/usr/bin/env python3
"""
Cloud Edge Module - Data transmission and cloud gateway
Handles MQTT/gRPC/QUIC transmission and Kafka producer
"""

import os
import json
import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from enum import Enum

try:
    from confluent_kafka import Producer, KafkaError
except ImportError:
    # Mock for development
    class Producer:
        def __init__(self, config):
            self.config = config
        def poll(self, timeout):
            pass
        def produce(self, topic, key=None, value=None, callback=None):
            if callback:
                callback(None, None)
        def flush(self, timeout=None):
            pass

try:
    import paho.mqtt.client as mqtt
except ImportError:
    # Mock for development
    class mqtt:
        class Client:
            def __init__(self, client_id):
                self.client_id = client_id
            def connect(self, host, port):
                pass
            def publish(self, topic, payload):
                pass
            def loop_start(self):
                pass
            def loop_stop(self):
                pass
            def disconnect(self):
                pass

try:
    import boto3
except ImportError:
    # Mock for development
    class MockS3:
        def __init__(self, *args, **kwargs):
            pass
        def upload_fileobj(self, fileobj, bucket, key):
            pass
    boto3 = type('boto3', (), {'client': MockS3})()

from config import Config


class TransmissionProtocol(Enum):
    """Data transmission protocol based on data size"""
    MQTT = "mqtt"
    HTTP2 = "http2"
    QUIC = "quic"


@dataclass
class TransmissionResult:
    """Result of data transmission"""
    success: bool
    protocol: str
    bytes_transferred: int
    duration_ms: float
    error_message: Optional[str] = None


class TransmissionRouter:
    """
    Routes data to appropriate transmission protocol based on size
    """

    def __init__(self):
        self.small_threshold = Config.Transmission.SMALL_PACKET_SIZE_KB * 1024  # bytes
        self.large_threshold = Config.Transmission.LARGE_PACKET_SIZE_MB * 1024 * 1024  # bytes

    def select_protocol(self, data_size: int) -> TransmissionProtocol:
        """
        Select transmission protocol based on data size

        Args:
            data_size: Size of data in bytes

        Returns:
            Appropriate transmission protocol
        """
        if data_size < self.small_threshold:
            return TransmissionProtocol.MQTT
        elif data_size < self.large_threshold:
            return TransmissionProtocol.HTTP2
        else:
            return TransmissionProtocol.QUIC

    def route(self, data: bytes, protocol: Optional[TransmissionProtocol] = None) -> Callable:
        """
        Route data to appropriate transmitter

        Args:
            data: Data to transmit
            protocol: Explicit protocol (auto-select if None)

        Returns:
            Function to perform transmission
        """
        if protocol is None:
            protocol = self.select_protocol(len(data))

        if protocol == TransmissionProtocol.MQTT:
            return self._transmit_mqtt
        elif protocol == TransmissionProtocol.HTTP2:
            return self._transmit_http2
        elif protocol == TransmissionProtocol.QUIC:
            return self._transmit_quic

        raise ValueError(f"Unknown protocol: {protocol}")

    def _transmit_mqtt(self, topic: str, data: bytes, **kwargs) -> TransmissionResult:
        """Transmit via MQTT"""
        start_time = time.time()
        try:
            # Implementation handled by CloudGateway
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(True, "MQTT", len(data), duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(False, "MQTT", 0, duration_ms, str(e))

    def _transmit_http2(self, endpoint: str, data: bytes, **kwargs) -> TransmissionResult:
        """Transmit via HTTP/2"""
        start_time = time.time()
        try:
            # Implementation handled by CloudGateway
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(True, "HTTP2", len(data), duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(False, "HTTP2", 0, duration_ms, str(e))

    def _transmit_quic(self, endpoint: str, data: bytes, **kwargs) -> TransmissionResult:
        """Transmit via QUIC"""
        start_time = time.time()
        try:
            # Implementation handled by CloudGateway
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(True, "QUIC", len(data), duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(False, "QUIC", 0, duration_ms, str(e))


class CertificateValidator:
    """
    Validates vehicle DID certificates
    """

    def __init__(self):
        self.authority_keys = set()  # Set of trusted authority public keys

    def validate_did(self, did: str, signature: str, payload: bytes) -> bool:
        """
        Validate vehicle DID certificate

        Args:
            did: Vehicle DID identifier
            signature: Signature payload
            payload: Signed data

        Returns:
            True if valid, False otherwise
        """
        # Simplified - in production would use proper crypto validation
        # Verify signature against authority keys
        return True  # Placeholder


class CloudGateway:
    """
    Cloud gateway for data ingress
    Handles MQTT/gRPC/QUIC protocols and certificate validation
    """

    def __init__(self):
        self.transmission_router = TransmissionRouter()
        self.certificate_validator = CertificateValidator()
        self.mqtt_client = None
        self._setup_mqtt()

    def _setup_mqtt(self):
        """Setup MQTT client"""
        self.mqtt_client = mqtt.Client(client_id=f"vehicle_{os.getpid()}")
        self.mqtt_client.on_connect = self._mqtt_on_connect
        self.mqtt_client.on_publish = self._mqtt_on_publish

    def _mqtt_on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        logging.info(f"MQTT connected with result code {rc}")

    def _mqtt_on_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        logging.info(f"MQTT message {mid} published")

    def connect(self):
        """Connect to cloud gateway"""
        try:
            self.mqtt_client.connect(
                Config.Transmission.MQTT_BROKER,
                Config.Transmission.MQTT_PORT,
                60
            )
            self.mqtt_client.loop_start()
            logging.info("Connected to MQTT broker")
        except Exception as e:
            logging.error(f"MQTT connection failed: {e}")

    def upload_metadata(self, metadata: Dict[str, Any], did: str, signature: str) -> bool:
        """
        Upload metadata via MQTT

        Args:
            metadata: Event metadata
            did: Vehicle DID
            signature: Certificate signature

        Returns:
            True if successful
        """
        # Validate certificate
        if not self.certificate_validator.validate_did(did, signature, json.dumps(metadata).encode()):
            logging.warning(f"Certificate validation failed for DID: {did}")
            return False

        # Select and execute transmission
        payload = json.dumps(metadata).encode()
        protocol = self.transmission_router.select_protocol(len(payload))
        transmitter = self.transmission_router.route(payload, protocol)

        if protocol == TransmissionProtocol.MQTT:
            try:
                topic = Config.Transmission.MQTT_TOPIC_METADATA
                self.mqtt_client.publish(topic, payload, qos=Config.Transmission.MQTT_QOS)
                logging.info(f"Metadata uploaded via MQTT to {topic}")
                return True
            except Exception as e:
                logging.error(f"MQTT upload failed: {e}")
                return False

        return False

    def upload_raw_data(self, data: bytes, file_path: str, did: str, signature: str) -> TransmissionResult:
        """
        Upload raw sensor data via appropriate protocol

        Args:
            data: Raw data bytes
            file_path: Target file path
            did: Vehicle DID
            signature: Certificate signature

        Returns:
            Transmission result
        """
        # Validate certificate
        if not self.certificate_validator.validate_did(did, signature, file_path.encode()):
            logging.warning(f"Certificate validation failed for DID: {did}")
            return TransmissionResult(False, "UNKNOWN", 0, 0, "Certificate validation failed")

        # Select and execute transmission
        protocol = self.transmission_router.select_protocol(len(data))
        transmitter = self.transmission_router.route(data, protocol)

        if protocol == TransmissionProtocol.HTTP2:
            return self._http2_upload(data, file_path)
        elif protocol == TransmissionProtocol.QUIC:
            return self._quic_upload(data, file_path)

        return TransmissionResult(False, "UNKNOWN", 0, 0, "No suitable protocol")

    def _http2_upload(self, data: bytes, file_path: str) -> TransmissionResult:
        """Upload via HTTP/2 with chunking"""
        start_time = time.time()
        try:
            endpoint = Config.Transmission.HTTP2_ENDPOINT
            chunk_size = Config.Transmission.UPLOAD_CHUNK_SIZE

            # Simulate chunked upload
            for offset in range(0, len(data), chunk_size):
                chunk = data[offset:offset + chunk_size]
                # In production: requests.post(endpoint, files={'chunk': chunk})
                time.sleep(0.01)  # Simulate network delay

            duration_ms = (time.time() - start_time) * 1000
            logging.info(f"HTTP/2 upload complete: {len(data)} bytes, {duration_ms:.0f}ms")
            return TransmissionResult(True, "HTTP2", len(data), duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(False, "HTTP2", 0, duration_ms, str(e))

    def _quic_upload(self, data: bytes, file_path: str) -> TransmissionResult:
        """Upload via QUIC protocol"""
        start_time = time.time()
        try:
            endpoint = Config.Transmission.QUIC_ENDPOINT
            # In production: aioquic or quic-go client
            duration_ms = (time.time() - start_time) * 1000
            logging.info(f"QUIC upload complete: {len(data)} bytes, {duration_ms:.0f}ms")
            return TransmissionResult(True, "QUIC", len(data), duration_ms)
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return TransmissionResult(False, "QUIC", 0, duration_ms, str(e))

    def disconnect(self):
        """Disconnect from cloud gateway"""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


class KafkaProducerWrapper:
    """
    Apache Kafka Producer wrapper
    Implements real producer->topic flow
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka producer

        Args:
            config: Custom configuration (uses Config.Kafka.PRODUCER_CONFIG if None)
        """
        if config is None:
            config = Config.Kafka.PRODUCER_CONFIG

        self.config = config
        self.producer = Producer(config)
        self.topic_metadata = Config.Kafka.TOPIC_METADATA
        self.topic_sensor_raw = Config.Kafka.TOPIC_SENSOR_RAW
        self.topic_status = Config.Kafka.TOPIC_STATUS
        self.topic_alert = Config.Kafka.TOPIC_ALERT

        # Callbacks for async delivery reports
        self.delivery_callbacks = {}

        logging.info(f"Kafka Producer initialized: {config['bootstrap.servers']}")

    def delivery_report(self, err, msg):
        """
        Callback for message delivery report

        Args:
            err: Error message (None if successful)
            msg: Message object
        """
        if err is not None:
            logging.error(f"Message delivery failed: {err}")
        else:
            logging.debug(f"Message delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}")

    def produce_metadata(self, event_id: str, metadata: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Produce metadata to Kafka topic

        Args:
            event_id: Unique event identifier
            metadata: Metadata dictionary
            key: Optional partition key (defaults to event_id)

        Returns:
            True if message was successfully produced
        """
        try:
            if key is None:
                key = event_id

            value = json.dumps(metadata, default=str)

            # Produce message to metadata topic
            self.producer.produce(
                self.topic_metadata,
                key=key.encode('utf-8'),
                value=value.encode('utf-8'),
                callback=self.delivery_report
            )

            # Flush to ensure message is sent
            self.producer.poll(0)

            logging.info(f"Produced metadata to {self.topic_metadata}: event_id={event_id}")
            return True

        except Exception as e:
            logging.error(f"Failed to produce metadata: {e}")
            return False

    def produce_sensor_raw(self, event_id: str, sensor_type: str,
                           sensor_data: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Produce raw sensor data to Kafka topic

        Args:
            event_id: Unique event identifier
            sensor_type: Type of sensor (camera, lidar, radar, etc.)
            sensor_data: Sensor data dictionary
            key: Optional partition key

        Returns:
            True if message was successfully produced
        """
        try:
            if key is None:
                key = event_id

            # Add metadata to sensor data
            payload = {
                "event_id": event_id,
                "sensor_type": sensor_type,
                "timestamp_ms": int(time.time() * 1000),
                **sensor_data
            }

            value = json.dumps(payload, default=str)

            # Produce message to sensor raw topic
            self.producer.produce(
                self.topic_sensor_raw,
                key=key.encode('utf-8'),
                value=value.encode('utf-8'),
                callback=self.delivery_report
            )

            self.producer.poll(0)

            logging.info(f"Produced sensor data to {self.topic_sensor_raw}: type={sensor_type}")
            return True

        except Exception as e:
            logging.error(f"Failed to produce sensor data: {e}")
            return False

    def produce_status(self, vehicle_id: str, status: Dict[str, Any], key: Optional[str] = None) -> bool:
        """
        Produce vehicle status to Kafka topic

        Args:
            vehicle_id: Vehicle identifier
            status: Status dictionary
            key: Optional partition key

        Returns:
            True if message was successfully produced
        """
        try:
            if key is None:
                key = vehicle_id

            payload = {
                "vehicle_id": vehicle_id,
                "timestamp_ms": int(time.time() * 1000),
                **status
            }

            value = json.dumps(payload, default=str)

            self.producer.produce(
                self.topic_status,
                key=key.encode('utf-8'),
                value=value.encode('utf-8'),
                callback=self.delivery_report
            )

            self.producer.poll(0)

            return True

        except Exception as e:
            logging.error(f"Failed to produce status: {e}")
            return False

    def produce_alert(self, event_id: str, alert_level: str,
                      alert_message: str, metadata: Dict[str, Any]) -> bool:
        """
        Produce alert to Kafka topic

        Args:
            event_id: Unique event identifier
            alert_level: Alert level (INFO, WARNING, CRITICAL)
            alert_message: Alert message
            metadata: Additional metadata

        Returns:
            True if message was successfully produced
        """
        try:
            payload = {
                "event_id": event_id,
                "alert_level": alert_level,
                "message": alert_message,
                "timestamp_ms": int(time.time() * 1000),
                **metadata
            }

            value = json.dumps(payload, default=str)

            self.producer.produce(
                self.topic_alert,
                key=event_id.encode('utf-8'),
                value=value.encode('utf-8'),
                callback=self.delivery_report
            )

            self.producer.poll(0)

            logging.warning(f"Produced alert: {alert_level} - {alert_message}")
            return True

        except Exception as e:
            logging.error(f"Failed to produce alert: {e}")
            return False

    def flush(self, timeout: float = 30.0):
        """
        Flush all pending messages

        Args:
            timeout: Timeout in seconds
        """
        logging.info("Flushing Kafka producer...")
        self.producer.flush(timeout=timeout)

    def close(self):
        """Close the producer"""
        self.flush()
        logging.info("Kafka Producer closed")


class StorageManager:
    """
    Manages tiered storage (S3/OSS, IoTDB, PostgreSQL)
    """

    def __init__(self):
        # S3 client for object storage
        self.s3_client = boto3.client(
            's3',
            endpoint_url=Config.Storage.S3_ENDPOINT,
            aws_access_key_id=Config.Storage.S3_ACCESS_KEY,
            aws_secret_access_key=Config.Storage.S3_SECRET_KEY
        )

        # Connection pools for databases (simulated)
        self.iotdb_pool = None
        self.pg_pool = None

    def store_raw_file(self, data: bytes, s3_path: str, metadata: Dict[str, Any] = None) -> bool:
        """
        Store raw file to S3/OSS

        Args:
            data: File data bytes
            s3_path: S3 path (s3://bucket/key)
            metadata: Optional metadata

        Returns:
            True if successful
        """
        try:
            # Parse S3 path
            if s3_path.startswith("s3://"):
                s3_path = s3_path[5:]
            parts = s3_path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            # Upload file
            from io import BytesIO
            self.s3_client.upload_fileobj(
                BytesIO(data),
                bucket,
                key,
                ExtraArgs={'Metadata': metadata} if metadata else None
            )

            logging.info(f"Stored file to S3: {bucket}/{key} ({len(data)} bytes)")
            return True

        except Exception as e:
            logging.error(f"Failed to store raw file: {e}")
            return False

    def store_can_signal(self, event_id: str, can_signals: list) -> bool:
        """
        Store CAN signals to time-series database (IoTDB)

        Args:
            event_id: Event identifier
            can_signals: List of CANSignal objects

        Returns:
            True if successful
        """
        try:
            # In production: Use IoTDB client
            # Simplified implementation
            logging.info(f"Stored {len(can_signals)} CAN signals for event {event_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to store CAN signals: {e}")
            return False

    def store_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Store metadata to PostgreSQL/Elasticsearch

        Args:
            metadata: Event metadata

        Returns:
            True if successful
        """
        try:
            # In production: Use PostgreSQL or Elasticsearch client
            logging.info(f"Stored metadata for event {metadata.get('event_id')}")
            return True
        except Exception as e:
            logging.error(f"Failed to store metadata: {e}")
            return False


class CloudEdgeOrchestrator:
    """
    Orchestrates cloud edge operations
    Coordinates gateway, Kafka producer, and storage
    """

    def __init__(self):
        self.gateway = CloudGateway()
        self.kafka_producer = KafkaProducerWrapper()
        self.storage_manager = StorageManager()

    def initialize(self):
        """Initialize all components"""
        self.gateway.connect()
        logging.info("Cloud Edge Orchestrator initialized")

    def upload_event(self, data_package: Dict[str, Any]) -> Dict[str, bool]:
        """
        Upload complete event to cloud

        Args:
            data_package: Complete data package from VehicleTriggerManager

        Returns:
            Dictionary of upload results
        """
        metadata = data_package["metadata"]
        event_id = metadata["event_id"]
        vehicle_id = metadata["vehicle_id"]

        results = {
            "metadata_kafka": False,
            "sensor_files_storage": False,
            "can_signals_tiered": False,
            "metadata_index": False,
        }

        # Step 1: Publish metadata to Kafka
        results["metadata_kafka"] = self.kafka_producer.produce_metadata(
            event_id, metadata
        )

        # Step 2: Upload sensor raw files to storage
        sensor_upload_success = True
        for file_info in metadata["file_manifest"]:
            # Extract file data from package
            file_data = self._extract_file_data(data_package, file_info["file_name"])
            if file_data is not None:
                success = self.storage_manager.store_raw_file(
                    file_data,
                    file_info["s3_path"],
                    {"event_id": event_id, "vehicle_id": vehicle_id}
                )
                sensor_upload_success = sensor_upload_success and success

        results["sensor_files_storage"] = sensor_upload_success

        # Step 3: Store CAN signals to tiered storage
        results["can_signals_tiered"] = self.storage_manager.store_can_signal(
            event_id, data_package["can_signals"]
        )

        # Step 4: Store metadata to index
        results["metadata_index"] = self.storage_manager.store_metadata(metadata)

        # Step 5: Publish status update
        self.kafka_producer.produce_status(
            vehicle_id,
            {
                "last_event_id": event_id,
                "upload_status": "complete" if all(results.values()) else "partial"
            }
        )

        return results

    def _extract_file_data(self, data_package: Dict[str, Any], filename: str) -> Optional[bytes]:
        """Extract file data from data package"""
        # In production, files would be read from disk or buffer
        # Simplified: return mock data
        if filename.endswith(".bag"):
            return b"mock_video_data"
        elif filename.endswith(".bin"):
            return b"mock_pointcloud_data"
        elif filename.endswith(".csv"):
            from vehicle_trigger import CANSignal
            can_csv = "timestamp_ms,vehicle_speed_kmh\n"
            for sig in data_package["can_signals"][:10]:
                can_csv += f"{sig.timestamp_ms},{sig.vehicle_speed_kmh:.1f}\n"
            return can_csv.encode()
        elif filename.endswith(".json"):
            return json.dumps(data_package["planning_decision"], default=str).encode()
        return None

    def shutdown(self):
        """Shutdown all components"""
        self.kafka_producer.flush()
        self.gateway.disconnect()
        logging.info("Cloud Edge Orchestrator shutdown complete")


if __name__ == '__main__':
    # Test cloud edge module
    logging.basicConfig(level=logging.INFO)
    print("Cloud Edge Module Test")
    print("=" * 50)

    orchestrator = CloudEdgeOrchestrator()
    orchestrator.initialize()

    # Create test data package
    test_metadata = {
        "event_id": "evt_test_001",
        "vehicle_id": "VIN_TEST_001",
        "trigger_type": "RULE_BASED",
        "trigger_time_utc": "2024-03-18T12:00:00Z",
        "location": {"latitude": 39.9042, "longitude": 116.4074},
        "scenario_tags": ["highway", "emergency_brake"],
        "trigger_details": {
            "rule_hit": "EMERGENCY_BRAKE",
            "perception_max_confidence": 0.75,
            "uncertainty_entropy": 0.3
        },
        "data_window": {
            "pre_trigger_seconds": 10,
            "post_trigger_seconds": 20,
            "total_duration_seconds": 30
        },
        "sensor_status": {
            "lidar_front": "OK",
            "camera_front_long": "OK",
            "radar_front": "OK",
            "gps_rtk": "FIXED",
            "imu": "OK"
        },
        "file_manifest": [
            {
                "file_name": "cam_front_long_00.bag",
                "type": "video_raw",
                "s3_path": f"s3://{Config.Storage.S3_BUCKET_RAW}/2024/03/18/VIN_TEST_001/evt_test_001/cam_front_long_00.bag"
            }
        ]
    }

    test_package = {
        "metadata": test_metadata,
        "can_signals": [],
        "sensor_data": [],
        "perception_output": None,
        "planning_decision": None,
        "shadow_decision": None,
    }

    # Upload test event
    results = orchestrator.upload_event(test_package)
    print(f"\nUpload Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    orchestrator.shutdown()
    print("\nTest completed!")
