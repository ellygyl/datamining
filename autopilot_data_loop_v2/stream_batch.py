#!/usr/bin/env python3
"""
Stream Batch Processing Module - Real-time and batch data processing
Implements Kafka consumer->group->consumer pattern, Flink stream processing, and Spark ETL
"""

import json
import time
import logging
import threading
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass

try:
    from confluent_kafka import Consumer, TopicPartition, KafkaError, KafkaException
except ImportError:
    # Mock for development
    class Consumer:
        def __init__(self, config):
            self.config = config
            self._committed = {}
        def subscribe(self, topics):
            pass
        def assign(self, partitions):
            pass
        def poll(self, timeout):
            return None
        def commit(self, offsets=None):
            pass
        def pause(self, partitions):
            pass
        def resume(self, partitions):
            pass
        def close(self):
            pass
    class TopicPartition:
        def __init__(self, topic, partition, offset=-1):
            self.topic = topic
            self.partition = partition
            self.offset = offset
    class KafkaError:
        _PARTITION_EOF = 0
    class KafkaException(Exception):
        pass

from config import Config


@dataclass
class StreamEvent:
    """Stream event from Kafka"""
    topic: str
    partition: int
    offset: int
    key: str
    value: Dict[str, Any]
    timestamp_ms: int


@dataclass
class WindowedData:
    """Data within a time window"""
    window_start_ms: int
    window_end_ms: int
    events: List[StreamEvent]
    aggregates: Dict[str, Any]


class KafkaConsumerGroup:
    """
    Apache Kafka Consumer Group
    Implements real consumer->topic->consumer group->consumer flow

    Consumer Group Architecture:
    - Multiple consumers belong to a group
    - Each partition is consumed by exactly one consumer in the group
    - Load balancing: partitions are distributed among consumers
    - Failover: if a consumer fails, partitions are reassigned
    """

    def __init__(self,
                 topics: List[str],
                 group_id: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka consumer group

        Args:
            topics: List of topics to subscribe to
            group_id: Consumer group ID (uses Config if None)
            config: Consumer configuration
        """
        if config is None:
            config = Config.Kafka.CONSUMER_CONFIG.copy()
        if group_id is not None:
            config['group.id'] = group_id

        self.config = config
        self.topics = topics
        self.group_id = config['group.id']

        # Create consumer
        self.consumer = Consumer(config)

        # Subscribe to topics (Consumer Group pattern)
        self.consumer.subscribe(topics)

        # Event handlers
        self.message_handlers: Dict[str, Callable[[StreamEvent], None]] = {}
        self.running = False
        self.consumer_thread: Optional[threading.Thread] = None

        logging.info(f"Kafka Consumer Group initialized: group={self.group_id}, topics={topics}")

    def register_handler(self, topic: str, handler: Callable[[StreamEvent], None]):
        """
        Register message handler for a topic

        Args:
            topic: Topic name
            handler: Handler function
        """
        self.message_handlers[topic] = handler
        logging.info(f"Handler registered for topic: {topic}")

    def start(self, blocking: bool = False):
        """
        Start consuming messages

        Args:
            blocking: If True, runs in main thread; if False, runs in background thread
        """
        self.running = True

        def consume_loop():
            while self.running:
                try:
                    # Poll for messages
                    msg = self.consumer.poll(timeout=1.0)

                    if msg is None:
                        continue

                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            # End of partition - not an error
                            continue
                        else:
                            logging.error(f"Kafka error: {msg.error()}")
                            continue

                    # Create StreamEvent
                    try:
                        value = json.loads(msg.value().decode('utf-8'))
                        event = StreamEvent(
                            topic=msg.topic(),
                            partition=msg.partition(),
                            offset=msg.offset(),
                            key=msg.key().decode('utf-8') if msg.key() else "",
                            value=value,
                            timestamp_ms=msg.timestamp()[1] if msg.timestamp()[0] != 0 else int(time.time() * 1000)
                        )

                        # Call registered handler
                        handler = self.message_handlers.get(msg.topic())
                        if handler:
                            handler(event)
                        else:
                            logging.warning(f"No handler for topic: {msg.topic()}")

                        # Auto-commit (if enabled in config)
                        if self.config.get('enable.auto.commit', True):
                            pass  # Consumer auto-commits

                    except Exception as e:
                        logging.error(f"Error processing message: {e}")

                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    logging.error(f"Consumer error: {e}")

        if blocking:
            consume_loop()
        else:
            self.consumer_thread = threading.Thread(target=consume_loop, daemon=True)
            self.consumer_thread.start()
            logging.info(f"Consumer group '{self.group_id}' started in background")

    def stop(self):
        """Stop consuming messages"""
        self.running = False
        if self.consumer_thread:
            self.consumer_thread.join(timeout=5.0)

        # Close consumer
        try:
            self.consumer.close()
            logging.info(f"Consumer group '{self.group_id}' stopped")
        except Exception as e:
            logging.error(f"Error closing consumer: {e}")

    def commit(self, offsets: Optional[Dict[Tuple[str, int], int]] = None):
        """
        Commit offsets manually

        Args:
            offsets: Dictionary of (topic, partition) -> offset
        """
        try:
            if offsets:
                tps = [TopicPartition(topic, partition, offset)
                       for (topic, partition), offset in offsets.items()]
                self.consumer.commit(offsets=tps)
            else:
                self.consumer.commit()
            logging.debug(f"Offsets committed")
        except Exception as e:
            logging.error(f"Error committing offsets: {e}")

    def pause(self, partitions: List[TopicPartition]):
        """Pause consumption of specific partitions"""
        self.consumer.pause(partitions)

    def resume(self, partitions: List[TopicPartition]):
        """Resume consumption of paused partitions"""
        self.consumer.resume(partitions)


class FlinkStreamProcessor:
    """
    Apache Flink-like Stream Processor
    Implements windowing, aggregation, and stateful processing

    Flink Processing Pipeline:
    Source (Kafka) -> KeyBy -> Window -> Process -> Sink (Alerts/Kafka)
    """

    def __init__(self, checkpoint_interval_ms: int = None):
        """
        Initialize Flink stream processor

        Args:
            checkpoint_interval_ms: Checkpoint interval (uses Config if None)
        """
        if checkpoint_interval_ms is None:
            checkpoint_interval_ms = Config.Flink.CHECKPOINTING_INTERVAL_MS

        self.checkpoint_interval_ms = checkpoint_interval_ms
        self.window_size_ms = Config.Flink.WINDOW_SIZE_SECONDS * 1000
        self.window_slide_ms = Config.Flink.WINDOW_SLIDE_SECONDS * 1000

        # Keyed state stores (simulating Flink KeyedState)
        self.keyed_state: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Operator state stores (simulating Flink OperatorState)
        self.operator_state: Dict[str, Any] = {}

        # Checkpoints
        self.checkpoints: List[Dict[str, Any]] = []
        self.last_checkpoint_time = time.time() * 1000

        # Window buffers
        self.window_buffers: Dict[str, List[StreamEvent]] = defaultdict(list)

        # Metrics
        self.metrics = {
            "events_processed": 0,
            "alerts_generated": 0,
            "windows_processed": 0
        }

        # Alert handlers
        self.alert_handlers: List[Callable[[Dict[str, Any]], None]] = []

        logging.info(f"FlinkStreamProcessor initialized: window={self.window_size_ms}ms, slide={self.window_slide_ms}ms")

    def register_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Register alert handler"""
        self.alert_handlers.append(handler)

    def process_event(self, event: StreamEvent):
        """
        Process a stream event through Flink pipeline

        Pipeline: Source -> KeyBy -> Window -> Process -> Sink
        """
        self.metrics["events_processed"] += 1

        # Step 1: KeyBy - partition by vehicle_id
        vehicle_id = event.value.get("vehicle_id", "unknown")
        key = f"vehicle:{vehicle_id}"

        # Step 2: Add to keyed state
        if "last_event_time" not in self.keyed_state[key]:
            self.keyed_state[key]["last_event_time"] = event.timestamp_ms
        self.keyed_state[key]["last_event_time"] = event.timestamp_ms
        self.keyed_state[key]["event_count"] = self.keyed_state[key].get("event_count", 0) + 1

        # Step 3: Add to window buffer
        self.window_buffers[key].append(event)

        # Step 4: Window processing - check if window is ready
        self._process_window(key, event.timestamp_ms)

        # Step 5: Periodic checkpoint
        current_time = time.time() * 1000
        if current_time - self.last_checkpoint_time > self.checkpoint_interval_ms:
            self._take_checkpoint()

    def _process_window(self, key: str, current_time_ms: int):
        """
        Process time window

        Implements tumbling window with aggregation
        """
        events = self.window_buffers[key]
        if not events:
            return

        # Find events within window
        window_start = current_time_ms - self.window_size_ms
        window_events = [e for e in events if e.timestamp_ms >= window_start]

        # Clean up old events
        self.window_buffers[key] = window_events

        if len(window_events) < 1:
            return

        # Calculate aggregates
        window_end = current_time_ms
        aggregates = self._calculate_aggregates(window_events)

        # Create windowed data
        windowed_data = WindowedData(
            window_start_ms=window_start,
            window_end_ms=window_end,
            events=window_events,
            aggregates=aggregates
        )

        # Process window - check for alerts
        self._process_window_alerts(key, windowed_data)

        self.metrics["windows_processed"] += 1

    def _calculate_aggregates(self, events: List[StreamEvent]) -> Dict[str, Any]:
        """Calculate aggregates for window"""
        aggregates = {
            "event_count": len(events),
            "vehicle_ids": set(),
            "trigger_types": defaultdict(int),
            "scenario_tags": defaultdict(int),
            "high_confidence_count": 0,
            "low_confidence_count": 0,
            "aeb_trigger_count": 0,
        }

        for event in events:
            value = event.value

            aggregates["vehicle_ids"].add(value.get("vehicle_id", "unknown"))

            trigger_type = value.get("trigger_type", "unknown")
            aggregates["trigger_types"][trigger_type] += 1

            # Count scenario tags
            for tag in value.get("scenario_tags", []):
                aggregates["scenario_tags"][tag] += 1

            # Count by confidence
            max_conf = value.get("trigger_details", {}).get("perception_max_confidence", 0.5)
            if max_conf > 0.7:
                aggregates["high_confidence_count"] += 1
            elif max_conf < 0.4:
                aggregates["low_confidence_count"] += 1

            # Count AEB triggers
            rule_hit = value.get("trigger_details", {}).get("rule_hit", "")
            if "AEB" in rule_hit or "EMERGENCY_BRAKE" in rule_hit:
                aggregates["aeb_trigger_count"] += 1

        # Convert sets to lists for JSON serialization
        aggregates["vehicle_ids"] = list(aggregates["vehicle_ids"])
        aggregates["trigger_types"] = dict(aggregates["trigger_types"])
        aggregates["scenario_tags"] = dict(aggregates["scenario_tags"])

        return aggregates

    def _process_window_alerts(self, key: str, windowed_data: WindowedData):
        """
        Process window data and generate alerts

        Alert conditions:
        1. AEB trigger count > threshold
        2. High-risk scenario count > threshold
        """
        aggregates = windowed_data.aggregates

        # Alert 1: Frequent AEB triggers
        if aggregates["aeb_trigger_count"] >= Config.Flink.AEB_TRIGGER_THRESHOLD_COUNT:
            alert = {
                "alert_type": "FREQUENT_AEB",
                "key": key,
                "window_start": windowed_data.window_start_ms,
                "window_end": windowed_data.window_end_ms,
                "aeb_count": aggregates["aeb_trigger_count"],
                "event_count": aggregates["event_count"],
                "vehicle_ids": aggregates["vehicle_ids"],
                "timestamp_ms": int(time.time() * 1000),
            }
            self._emit_alert(alert)

        # Alert 2: High-risk scenarios
        high_risk_tags = ["construction_zone", "low_visibility", "human_takeover", "obstacle_avoidance"]
        high_risk_count = sum(aggregates["scenario_tags"].get(tag, 0) for tag in high_risk_tags)

        if high_risk_count >= Config.Flink.HIGH_RISK_SCENARIO_THRESHOLD:
            alert = {
                "alert_type": "HIGH_RISK_SCENARIOS",
                "key": key,
                "window_start": windowed_data.window_start_ms,
                "window_end": windowed_data.window_end_ms,
                "risk_count": high_risk_count,
                "scenario_tags": aggregates["scenario_tags"],
                "vehicle_ids": aggregates["vehicle_ids"],
                "timestamp_ms": int(time.time() * 1000),
            }
            self._emit_alert(alert)

    def _emit_alert(self, alert: Dict[str, Any]):
        """Emit alert to registered handlers"""
        self.metrics["alerts_generated"] += 1
        logging.warning(f"Alert emitted: {alert['alert_type']}")

        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Error in alert handler: {e}")

    def _take_checkpoint(self):
        """Take a checkpoint of state"""
        self.last_checkpoint_time = time.time() * 1000

        checkpoint = {
            "checkpoint_id": len(self.checkpoints),
            "timestamp_ms": int(self.last_checkpoint_time),
            "keyed_state": {k: dict(v) for k, v in self.keyed_state.items()},
            "operator_state": dict(self.operator_state),
            "metrics": dict(self.metrics),
        }

        self.checkpoints.append(checkpoint)

        # Keep only last 5 checkpoints
        if len(self.checkpoints) > 5:
            self.checkpoints.pop(0)

        logging.debug(f"Checkpoint taken: {checkpoint['checkpoint_id']}")

    def restore_from_checkpoint(self, checkpoint_id: int):
        """Restore state from checkpoint"""
        for checkpoint in self.checkpoints:
            if checkpoint["checkpoint_id"] == checkpoint_id:
                self.keyed_state = defaultdict(dict, {
                    k: defaultdict(dict, v) for k, v in checkpoint["keyed_state"].items()
                })
                self.operator_state = defaultdict(dict, checkpoint["operator_state"])
                self.metrics = checkpoint["metrics"].copy()
                logging.info(f"Restored from checkpoint {checkpoint_id}")
                return

        logging.warning(f"Checkpoint {checkpoint_id} not found")

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for vehicle health dashboard"""
        return {
            "vehicles_active": len(self.keyed_state),
            "total_events": self.metrics["events_processed"],
            "total_alerts": self.metrics["alerts_generated"],
            "total_windows": self.metrics["windows_processed"],
            "recent_alerts": [
                {
                    "type": alert.get("alert_type"),
                    "timestamp": alert.get("timestamp_ms"),
                }
                for alert in self._get_recent_alerts(10)
            ]
        }

    def _get_recent_alerts(self, count: int) -> List[Dict[str, Any]]:
        """Get recent alerts (would be stored in production)"""
        return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        return dict(self.metrics)


class SparkETLProcessor:
    """
    Apache Spark Batch ETL Processor
    Implements data cleaning, validation, and transformation

    ETL Pipeline:
    Extract (S3/DB) -> Transform (Clean, Validate, Sync) -> Load (Processed DB)
    """

    def __init__(self):
        """Initialize Spark ETL processor"""
        self.etl_jobs = []
        self.validation_rules = self._init_validation_rules()
        self.transformations = self._init_transformations()

        logging.info("SparkETLProcessor initialized")

    def _init_validation_rules(self) -> Dict[str, Callable]:
        """Initialize data validation rules"""
        return {
            "timestamp_sync": self._validate_timestamp_sync,
            "sensor_health": self._validate_sensor_health,
            "data_completeness": self._validate_data_completeness,
            "gps_validity": self._validate_gps_validity,
        }

    def _init_transformations(self) -> Dict[str, Callable]:
        """Initialize data transformations"""
        return {
            "time_sync": self._transform_time_sync,
            "sensor_calibration": self._transform_sensor_calibration,
            "coordinate_projection": self._transform_coordinate_projection,
            "feature_extraction": self._transform_feature_extraction,
        }

    def run_etl_job(self, date_range: tuple[datetime, datetime],
                    scenario_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Run ETL job for a date range

        Args:
            date_range: (start_date, end_date)
            scenario_filter: Optional scenario type filter

        Returns:
            ETL job results
        """
        start_date, end_date = date_range

        job_id = f"etl_job_{int(time.time())}"
        logging.info(f"Starting ETL job {job_id} for {start_date} to {end_date}")

        job_results = {
            "job_id": job_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "status": "running",
            "extracted": 0,
            "validated": 0,
            "transformed": 0,
            "loaded": 0,
            "errors": []
        }

        try:
            # Phase 1: Extract
            extracted_events = self._extract_data(start_date, end_date, scenario_filter)
            job_results["extracted"] = len(extracted_events)

            # Phase 2: Validate
            validated_events = self._validate_data(extracted_events)
            job_results["validated"] = len(validated_events)

            # Phase 3: Transform
            transformed_events = self._transform_data(validated_events)
            job_results["transformed"] = len(transformed_events)

            # Phase 4: Load
            loaded_count = self._load_data(transformed_events)
            job_results["loaded"] = loaded_count

            job_results["status"] = "completed"
            logging.info(f"ETL job {job_id} completed: {job_results}")

        except Exception as e:
            job_results["status"] = "failed"
            job_results["errors"].append(str(e))
            logging.error(f"ETL job {job_id} failed: {e}")

        return job_results

    def _extract_data(self, start_date: datetime, end_date: datetime,
                      scenario_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract data from raw storage

        In production: Query S3, IoTDB, PostgreSQL
        """
        # Simulated extraction
        events = []

        # Generate mock events for testing
        current_date = start_date
        event_count = 0

        while current_date <= end_date and event_count < 1000:
            event = self._generate_mock_event(current_date)
            if scenario_filter is None or scenario_filter in event.get("scenario_tags", []):
                events.append(event)
                event_count += 1
            current_date += timedelta(hours=1)

        logging.info(f"Extracted {len(events)} events")
        return events

    def _validate_data(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate data using all validation rules

        Returns only events that pass all validations
        """
        validated = []
        validation_errors = 0

        for event in events:
            valid = True
            errors = []

            for rule_name, rule_func in self.validation_rules.items():
                try:
                    is_valid, error_msg = rule_func(event)
                    if not is_valid:
                        valid = False
                        errors.append(f"{rule_name}: {error_msg}")
                except Exception as e:
                    valid = False
                    errors.append(f"{rule_name}: {str(e)}")

            if valid:
                validated.append(event)
            else:
                validation_errors += 1
                logging.debug(f"Event {event.get('event_id')} failed validation: {errors}")

        logging.info(f"Validated {len(validated)}/{len(events)} events ({validation_errors} failed)")
        return validated

    def _transform_data(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform data using all transformations
        """
        transformed = []

        for event in events:
            try:
                # Apply transformations in sequence
                transformed_event = event.copy()

                for transform_name, transform_func in self.transformations.items():
                    transformed_event = transform_func(transformed_event)

                transformed.append(transformed_event)
            except Exception as e:
                logging.warning(f"Failed to transform event {event.get('event_id')}: {e}")

        logging.info(f"Transformed {len(transformed)} events")
        return transformed

    def _load_data(self, events: List[Dict[str, Any]]) -> int:
        """
        Load processed data to destination

        In production: Write to processed S3 bucket, processed database
        """
        # Simulated load
        logging.info(f"Loaded {len(events)} events to processed storage")
        return len(events)

    # Validation rule implementations
    def _validate_timestamp_sync(self, event: Dict[str, Any]) -> tuple[bool, str]:
        """Validate timestamp synchronization between sensors"""
        # Check if timestamps within tolerance
        tolerance_ms = Config.Spark.TIME_SYNC_TOLERANCE_MS
        # Simulated check
        return True, ""

    def _validate_sensor_health(self, event: Dict[str, Any]) -> tuple[bool, str]:
        """Validate sensor health status"""
        sensor_status = event.get("sensor_status", {})
        # All sensors should be OK or FIXED
        for sensor, status in sensor_status.items():
            if status not in ["OK", "FIXED", "WARNING"]:
                return False, f"Sensor {sensor} has status {status}"
        return True, ""

    def _validate_data_completeness(self, event: Dict[str, Any]) -> tuple[bool, str]:
        """Validate data completeness"""
        required_fields = ["event_id", "vehicle_id", "trigger_type", "location", "file_manifest"]
        for field in required_fields:
            if field not in event:
                return False, f"Missing required field: {field}"
        return True, ""

    def _validate_gps_validity(self, event: Dict[str, Any]) -> tuple[bool, str]:
        """Validate GPS coordinates"""
        location = event.get("location", {})
        lat = location.get("latitude", 0)
        lon = location.get("longitude", 0)

        if not (-90 <= lat <= 90):
            return False, f"Invalid latitude: {lat}"
        if not (-180 <= lon <= 180):
            return False, f"Invalid longitude: {lon}"

        return True, ""

    # Transformation implementations
    def _transform_time_sync(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Transform: Synchronize sensor timestamps"""
        event["time_sync_applied"] = True
        return event

    def _transform_sensor_calibration(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Transform: Apply sensor calibration"""
        event["calibration_applied"] = True
        return event

    def _transform_coordinate_projection(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Transform: Project coordinates to local frame"""
        event["coordinate_projection_applied"] = True
        return event

    def _transform_feature_extraction(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Transform: Extract features for ML"""
        event["features"] = {
            "speed_range": event.get("trigger_details", {}).get("speed_range", 0),
            "acceleration": event.get("trigger_details", {}).get("acceleration", 0),
            "confidence": event.get("trigger_details", {}).get("perception_max_confidence", 0),
        }
        return event

    def _generate_mock_event(self, date: datetime) -> Dict[str, Any]:
        """Generate mock event for testing"""
        return {
            "event_id": f"evt_{int(date.timestamp())}",
            "vehicle_id": f"VIN_{int(date.timestamp()) % 100:03d}",
            "trigger_type": "RULE_BASED",
            "trigger_time_utc": date.isoformat(),
            "location": {
                "latitude": 39.9042,
                "longitude": 116.4074,
                "altitude": 52.5,
            },
            "scenario_tags": ["highway"],
            "trigger_details": {
                "perception_max_confidence": 0.75,
            },
            "sensor_status": {
                "lidar_front": "OK",
                "camera_front_long": "OK",
                "radar_front": "OK",
                "gps_rtk": "FIXED",
                "imu": "OK"
            },
            "file_manifest": [],
        }


class StreamBatchOrchestrator:
    """
    Orchestrates stream and batch processing
    Coordinates Kafka consumer, Flink processor, and Spark ETL
    """

    def __init__(self):
        self.consumer_group = None
        self.flink_processor = FlinkStreamProcessor()
        self.spark_processor = SparkETLProcessor()
        self.alert_messages: List[Dict[str, Any]] = []

    def start_stream_processing(self):
        """Start real-time stream processing"""
        # Create Kafka consumer group for metadata topic
        self.consumer_group = KafkaConsumerGroup(
            topics=[Config.Kafka.TOPIC_METADATA,
                     Config.Kafka.TOPIC_ALERT,
                     Config.Kafka.TOPIC_STATUS],
            group_id=Config.Kafka.CONSUMER_CONFIG['group.id']
        )

        # Register handlers
        self.consumer_group.register_handler(
            Config.Kafka.TOPIC_METADATA,
            self._handle_metadata_event
        )
        self.consumer_group.register_handler(
            Config.Kafka.TOPIC_ALERT,
            self._handle_alert_event
        )
        self.consumer_group.register_handler(
            Config.Kafka.TOPIC_STATUS,
            self._handle_status_event
        )

        # Register alert handler with Flink
        self.flink_processor.register_alert_handler(self._handle_flink_alert)

        # Start consumer in background
        self.consumer_group.start(blocking=False)

        logging.info("Stream processing started")

    def _handle_metadata_event(self, event: StreamEvent):
        """Handle metadata event from Kafka"""
        # Process through Flink pipeline
        self.flink_processor.process_event(event)

    def _handle_alert_event(self, event: StreamEvent):
        """Handle alert event from Kafka"""
        self.alert_messages.append(event.value)

    def _handle_status_event(self, event: StreamEvent):
        """Handle status event from Kafka"""
        # Update vehicle status tracking
        pass

    def _handle_flink_alert(self, alert: Dict[str, Any]):
        """Handle alert from Flink processor"""
        logging.warning(f"Flink Alert: {alert}")

    def run_batch_etl(self, days_back: int = 1) -> Dict[str, Any]:
        """
        Run batch ETL job

        Args:
            days_back: Number of days to process
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        return self.spark_processor.run_etl_job((start_date, end_date))

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return {
            "stream_metrics": self.flink_processor.get_metrics(),
            "dashboard": self.flink_processor.get_dashboard_data(),
            "recent_alerts": self.alert_messages[-10:] if self.alert_messages else [],
        }

    def shutdown(self):
        """Shutdown all processors"""
        if self.consumer_group:
            self.consumer_group.stop()
        logging.info("StreamBatchOrchestrator shutdown complete")


if __name__ == '__main__':
    # Test stream batch module
    logging.basicConfig(level=logging.INFO)
    print("Stream Batch Processing Module Test")
    print("=" * 50)

    orchestrator = StreamBatchOrchestrator()

    # Test Spark ETL
    print("\nTesting Spark ETL...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    etl_results = orchestrator.spark_processor.run_etl_job((start_date, end_date))
    print(f"ETL Results: {etl_results}")

    # Test Flink processor
    print("\nTesting Flink Stream Processor...")
    from stream_batch import StreamEvent

    test_events = [
        StreamEvent(
            topic="autopilot.metadata",
            partition=0,
            offset=0,
            key="evt_001",
            value={
                "event_id": "evt_001",
                "vehicle_id": "VIN_001",
                "trigger_type": "RULE_BASED",
                "trigger_details": {
                    "rule_hit": "EMERGENCY_BRAKE",
                    "perception_max_confidence": 0.75
                },
                "scenario_tags": ["highway"]
            },
            timestamp_ms=int(time.time() * 1000)
        ),
        StreamEvent(
            topic="autopilot.metadata",
            partition=0,
            offset=1,
            key="evt_002",
            value={
                "event_id": "evt_002",
                "vehicle_id": "VIN_001",
                "trigger_type": "MODEL_BASED",
                "trigger_details": {
                    "rule_hit": "LOW_CONFIDENCE",
                    "perception_max_confidence": 0.3
                },
                "scenario_tags": ["construction_zone"]
            },
            timestamp_ms=int(time.time() * 1000)
        ),
    ]

    for event in test_events:
        orchestrator.flink_processor.process_event(event)

    print(f"Flink Metrics: {orchestrator.flink_processor.get_metrics()}")
    print(f"Dashboard Data: {orchestrator.flink_processor.get_dashboard_data()}")

    print("\nTest completed!")
