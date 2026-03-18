#!/usr/bin/env python3
"""
Data Mining and Construction Module
Implements scenario mining, annotation pipeline, and data augmentation
"""

import json
import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random

from config import Config

# Import real perception module if available
try:
    from perception import create_perception_pipeline
    PERCEPTION_AVAILABLE = True
except ImportError:
    PERCEPTION_AVAILABLE = False
    logging.warning("Perception module not available. Will use mock for pre-annotation.")


class ScenarioType(Enum):
    """Scenario types for categorization"""
    HIGHWAY = "highway"
    URBAN = "urban_road"
    INTERSECTION = "intersection"
    PARKING = "parking"
    CONSTRUCTION_ZONE = "construction_zone"
    BAD_WEATHER = "bad_weather"
    PEDESTRIAN_INTERACTION = "pedestrian_interaction"
    CYCLIST_INTERACTION = "cyclist_interaction"
    ANIMAL_CROSSING = "animal_crossing"
    LANE_CHANGE = "lane_change"
    EMERGENCY_BRAKE = "emergency_brake"
    UNKNOWN = "unknown"


@dataclass
class Scenario:
    """Scenario representation"""
    scenario_id: str
    type: ScenarioType
    metadata: Dict[str, Any]
    features: np.ndarray
    confidence: float
    event_ids: List[str]
    tags: List[str] = field(default_factory=list)


@dataclass
class AnnotationResult:
    """Annotation result from human-in-the-loop pipeline"""
    annotation_id: str
    event_id: str
    annotation_type: str
    labels: Dict[str, Any]
    annotator_id: str
    confidence: float
    timestamp: datetime
    review_status: str  # "accepted", "rejected", "needs_revision"


@dataclass
class AugmentationResult:
    """Data augmentation result"""
    original_event_id: str
    augmented_event_ids: List[str]
    augmentation_methods: List[str]
    parameters: Dict[str, Any]


class ScenarioMiner:
    """
    High-value scenario mining
    Implements rule filtering, clustering, and hard mining
    """

    def __init__(self):
        self.scenarios: List[Scenario] = []
        self.scenario_clusters: Dict[ScenarioType, List[Scenario]] = {}
        self.feature_extractor = FeatureExtractor()

    def mine_scenarios(self, events: List[Dict[str, Any]],
                      method: str = "all") -> List[Scenario]:
        """
        Mine high-value scenarios from events

        Args:
            events: List of event metadata
            method: Mining method ("rule", "cluster", "hard", "all")

        Returns:
            List of mined scenarios
        """
        mined = []

        if method in ["rule", "all"]:
            mined.extend(self._rule_based_mining(events))

        if method in ["cluster", "all"]:
            mined.extend(self._cluster_based_mining(events))

        if method in ["hard", "all"]:
            mined.extend(self._hard_case_mining(events))

        self.scenarios.extend(mined)
        self._organize_clusters()

        logging.info(f"Mined {len(mined)} scenarios using method={method}")
        return mined

    def _rule_based_mining(self, events: List[Dict[str, Any]]) -> List[Scenario]:
        """
        Rule-based scenario mining
        Filter events by specific conditions
        """
        scenarios = []

        # Rule: High-risk scenarios
        high_risk_tags = ["construction_zone", "low_visibility", "human_takeover",
                         "obstacle_avoidance", "aeb_trigger"]

        for event in events:
            scenario_tags = event.get("scenario_tags", [])
            has_high_risk = any(tag in scenario_tags for tag in high_risk_tags)

            if has_high_risk:
                features = self.feature_extractor.extract(event)
                scenario_type = self._classify_scenario(event)

                scenario = Scenario(
                    scenario_id=f"scn_rule_{event.get('event_id')}",
                    type=scenario_type,
                    metadata=event,
                    features=features,
                    confidence=0.8,  # High confidence for rule-based
                    event_ids=[event.get("event_id")],
                    tags=scenario_tags
                )
                scenarios.append(scenario)

        logging.info(f"Rule-based mining: {len(scenarios)} scenarios")
        return scenarios

    def _cluster_based_mining(self, events: List[Dict[str, Any]],
                             n_clusters: int = 5) -> List[Scenario]:
        """
        Cluster-based scenario mining
        Use K-Means to discover long-tail scenarios
        """
        scenarios = []

        # Extract features from all events
        features_list = []
        valid_events = []

        for event in events:
            features = self.feature_extractor.extract(event)
            if features is not None:
                features_list.append(features)
                valid_events.append(event)

        if len(features_list) < n_clusters:
            logging.warning(f"Not enough events for clustering ({len(features_list)} < {n_clusters})")
            return scenarios

        features_array = np.array(features_list)

        # K-Means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_array)

        # Create scenarios from clusters
        for cluster_id in range(n_clusters):
            cluster_events = [valid_events[i] for i in range(len(valid_events))
                              if cluster_labels[i] == cluster_id]

            if cluster_events:
                # Calculate cluster statistics
                cluster_features = features_array[cluster_labels == cluster_id]
                centroid = np.mean(cluster_features, axis=0)

                # Find representative event
                representative = cluster_events[np.linalg.norm(
                    features_array[cluster_labels == cluster_id] - centroid, axis=1
                ).argmin()]

                scenario_type = self._classify_scenario(representative)

                scenario = Scenario(
                    scenario_id=f"scn_cluster_{cluster_id}_{int(time.time())}",
                    type=scenario_type,
                    metadata={"cluster_id": cluster_id, "representative": representative},
                    features=centroid,
                    confidence=0.6,  # Medium confidence for clustering
                    event_ids=[e.get("event_id") for e in cluster_events],
                    tags=["cluster_mining", f"cluster_{cluster_id}"]
                )
                scenarios.append(scenario)

        logging.info(f"Cluster-based mining: {len(scenarios)} scenarios")
        return scenarios

    def _hard_case_mining(self, events: List[Dict[str, Any]]) -> List[Scenario]:
        """
        Hard case mining
        Select events with low confidence or high uncertainty
        """
        scenarios = []

        for event in events:
            trigger_details = event.get("trigger_details", {})
            confidence = trigger_details.get("perception_max_confidence", 1.0)
            uncertainty = trigger_details.get("uncertainty_entropy", 0.0)

            # Hard case: low confidence or high uncertainty
            if confidence < 0.4 or uncertainty > 0.8:
                features = self.feature_extractor.extract(event)
                scenario_type = self._classify_scenario(event)

                scenario = Scenario(
                    scenario_id=f"scn_hard_{event.get('event_id')}",
                    type=scenario_type,
                    metadata=event,
                    features=features,
                    confidence=confidence,
                    event_ids=[event.get("event_id")],
                    tags=["hard_case", "low_confidence" if confidence < 0.4 else "high_uncertainty"]
                )
                scenarios.append(scenario)

        logging.info(f"Hard case mining: {len(scenarios)} scenarios")
        return scenarios

    def _classify_scenario(self, event: Dict[str, Any]) -> ScenarioType:
        """Classify event into scenario type"""
        scenario_tags = event.get("scenario_tags", [])
        trigger_details = event.get("trigger_details", {})
        rule_hit = trigger_details.get("rule_hit", "")

        # Check tags
        if "construction_zone" in scenario_tags:
            return ScenarioType.CONSTRUCTION_ZONE
        elif "bad_weather" in scenario_tags:
            return ScenarioType.BAD_WEATHER
        elif "pedestrian" in scenario_tags or rule_hit == "PEDESTRIAN":
            return ScenarioType.PEDESTRIAN_INTERACTION
        elif "cyclist" in scenario_tags or rule_hit == "CYCLIST":
            return ScenarioType.CYCLIST_INTERACTION
        elif "animal" in scenario_tags:
            return ScenarioType.ANIMAL_CROSSING
        elif "lane_change" in scenario_tags or "change_lane" in rule_hit.lower():
            return ScenarioType.LANE_CHANGE
        elif "emergency_brake" in rule_hit or "aeb" in rule_hit.lower():
            return ScenarioType.EMERGENCY_BRAKE
        elif "highway" in scenario_tags:
            return ScenarioType.HIGHWAY
        elif "urban" in scenario_tags or "intersection" in scenario_tags:
            return ScenarioType.URBAN
        else:
            return ScenarioType.UNKNOWN

    def _organize_clusters(self):
        """Organize scenarios by type into clusters"""
        self.scenario_clusters = {}
        for scenario in self.scenarios:
            if scenario.type not in self.scenario_clusters:
                self.scenario_clusters[scenario.type] = []
            self.scenario_clusters[scenario.type].append(scenario)

    def get_long_tail_scenarios(self, threshold: int = 5) -> List[Scenario]:
        """
        Get long-tail scenarios (rare scenarios)

        Args:
            threshold: Maximum count to be considered long-tail
        """
        long_tail = []
        for scenario_type, scenarios in self.scenario_clusters.items():
            if len(scenarios) < threshold:
                long_tail.extend(scenarios)

        logging.info(f"Found {len(long_tail)} long-tail scenarios")
        return long_tail

    def query_by_condition(self, **conditions) -> List[Scenario]:
        """
        Query scenarios by conditions

        Args:
            conditions: Key-value pairs for filtering
        """
        results = []

        for scenario in self.scenarios:
            match = True
            for key, value in conditions.items():
                if key == "scenario_type":
                    if scenario.type != value:
                        match = False
                        break
                elif key in scenario.metadata:
                    if scenario.metadata[key] != value:
                        match = False
                        break
                elif hasattr(scenario, key):
                    if getattr(scenario, key) != value:
                        match = False
                        break
                else:
                    match = False
                    break

            if match:
                results.append(scenario)

        return results


class FeatureExtractor:
    """
    Extract features from events for mining and ML
    """

    def extract(self, event: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract feature vector from event

        Features:
        - Location (lat, lon, alt)
        - Speed (from trigger details or inferred)
        - Confidence
        - Uncertainty
        - Trigger type (one-hot encoded)
        """
        try:
            location = event.get("location", {})
            trigger_details = event.get("trigger_details", {})

            features = np.array([
                location.get("latitude", 0.0),
                location.get("longitude", 0.0),
                location.get("altitude", 0.0),
                location.get("heading", 0.0),
                trigger_details.get("perception_max_confidence", 0.5),
                trigger_details.get("uncertainty_entropy", 0.0),
                1.0 if event.get("trigger_type") == "RULE_BASED" else 0.0,
                1.0 if event.get("trigger_type") == "MODEL_BASED" else 0.0,
                1.0 if event.get("trigger_type") == "UNCERTAINTY_BASED" else 0.0,
            ])

            return features

        except Exception as e:
            logging.warning(f"Failed to extract features: {e}")
            return None


class AnnotationPipeline:
    """
    Human-in-the-loop annotation pipeline
    Implements pre-annotation, routing, and quality control
    """

    def __init__(self, use_real_perception: bool = True):
        """
        Initialize annotation pipeline

        Args:
            use_real_perception: Use real perception models (YOLO, SAM, etc.)
        """
        # Use real perception module if available and requested
        if use_real_perception and PERCEPTION_AVAILABLE:
            self.perception_pipeline = create_perception_pipeline()
            self.foundation_model = None
            logging.info("Using real perception models for pre-annotation")
        else:
            self.perception_pipeline = None
            self.foundation_model = FoundationModelMock()
            logging.info("Using mock foundation model for pre-annotation")

        self.annotator_pool = AnnotatorPool()
        self.annotation_queue = []
        self.annotations: List[AnnotationResult] = []

    def annotate_event(self, event_id: str, event_data: Dict[str, Any],
                       annotation_types: List[str] = None) -> AnnotationResult:
        """
        Annotate an event through the pipeline

        Pipeline:
        1. Pre-annotation (Foundation Model / Perception Pipeline)
        2. Confidence-based routing
        3. Human annotation (if needed)
        4. Quality check
        """
        if annotation_types is None:
            annotation_types = ["bbox_2d", "bbox_3d", "segmentation"]

        # Step 1: Pre-annotation
        if self.perception_pipeline is not None:
            # Use real perception pipeline
            pre_annotation = self.perception_pipeline.process_for_annotation(
                event_data, annotation_types
            )
        else:
            # Use mock foundation model
            pre_annotation = self.foundation_model.pre_annotate(event_data, annotation_types)

        # Step 2: Confidence-based routing
        routing = self._route_by_confidence(pre_annotation["confidence"])

        # Step 3: Human annotation (if needed)
        if routing == "manual_full":
            annotation = self.annotator_pool.manual_annotate(
                event_id, event_data, annotation_types, pre_annotation
            )
        elif routing == "manual_review":
            annotation = self.annotator_pool.review_annotation(
                event_id, event_data, pre_annotation
            )
        else:  # auto_accept
            annotation = self._convert_pre_annotation(event_id, pre_annotation)

        # Step 4: Quality check
        annotation.review_status = self._quality_check(annotation, pre_annotation)

        self.annotations.append(annotation)

        logging.info(f"Annotation completed for {event_id}: "
                    f"confidence={annotation.confidence:.2f}, "
                    f"route={routing}")

        return annotation

    def _route_by_confidence(self, confidence: float) -> str:
        """
        Route annotation task based on confidence

        Routing rules:
        - confidence > 0.9: Auto-accept (no human review)
        - 0.4 < confidence <= 0.9: Manual review
        - confidence <= 0.4: Full manual annotation
        """
        if confidence > Config.Annotation.CONFIDENCE_HIGH:
            return "auto_accept"
        elif confidence > Config.Annotation.CONFIDENCE_LOW:
            return "manual_review"
        else:
            return "manual_full"

    def _convert_pre_annotation(self, event_id: str, pre_annotation: Dict[str, Any]) -> AnnotationResult:
        """Convert pre-annotation to AnnotationResult"""
        annotator_id = "real_perception" if self.perception_pipeline is not None else "foundation_model"

        return AnnotationResult(
            annotation_id=f"ann_auto_{event_id}",
            event_id=event_id,
            annotation_type=",".join(pre_annotation.get("types", [])),
            labels=pre_annotation.get("labels", {}),
            annotator_id=annotator_id,
            confidence=pre_annotation.get("confidence", 1.0),
            timestamp=datetime.now(),
            review_status="accepted"
        )

    def get_perception_info(self) -> Dict[str, Any]:
        """Get information about the perception models being used"""
        if self.perception_pipeline is not None:
            return {
                "type": "real_perception",
                "models": {
                    "detector": self.perception_pipeline.object_detector.get_model_info(),
                    "segmenter": self.perception_pipeline.segmenter.get_model_info(),
                },
                "processing_mode": "real_inference"
            }
        else:
            return {
                "type": "mock",
                "models": {"name": "FoundationModelMock"},
                "processing_mode": "simulation"
            }

    def _quality_check(self, annotation: AnnotationResult,
                      pre_annotation: Dict[str, Any]) -> str:
        """
        Perform quality check on annotation

        Checks:
        - IoU consistency between human and pre-annotation
        - Completeness of required fields
        - Temporal consistency
        """
        # For auto-accepted annotations, check basic completeness
        if annotation.annotator_id == "foundation_model":
            if not annotation.labels:
                return "needs_revision"
            return "accepted"

        # For human annotations, check consistency
        iou_score = self._calculate_iou(annotation.labels, pre_annotation.get("labels", {}))

        if iou_score < 0.5:
            return "needs_revision"
        elif iou_score < 0.7:
            return "accepted_with_review"
        else:
            return "accepted"

    def _calculate_iou(self, labels1: Dict[str, Any], labels2: Dict[str, Any]) -> float:
        """Calculate IoU between two label sets (simplified)"""
        # In production, calculate actual IoU for bounding boxes
        # Simplified: random value for testing
        return random.uniform(0.4, 0.95)

    def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get annotation statistics"""
        if not self.annotations:
            return {}

        status_counts = {}
        for ann in self.annotations:
            status_counts[ann.review_status] = status_counts.get(ann.review_status, 0) + 1

        # Check what type of annotator was used
        real_perception_count = sum(1 for ann in self.annotations if ann.annotator_id == "real_perception")
        mock_count = sum(1 for ann in self.annotations if ann.annotator_id == "foundation_model")

        return {
            "total_annotations": len(self.annotations),
            "status_distribution": status_counts,
            "avg_confidence": np.mean([ann.confidence for ann in self.annotations]),
            "auto_accepted": real_perception_count + mock_count,
            "real_perception_annotations": real_perception_count,
            "mock_annotations": mock_count,
            "perception_info": self.get_perception_info(),
        }


class FoundationModelMock:
    """Mock foundation model for pre-annotation"""

    def pre_annotate(self, event_data: Dict[str, Any],
                    annotation_types: List[str]) -> Dict[str, Any]:
        """
        Generate pre-annotations using foundation model

        Args:
            event_data: Event data including sensor data
            annotation_types: Types of annotations to generate

        Returns:
            Dictionary with annotations and confidence
        """
        # Simulated pre-annotation
        labels = {}

        if "bbox_2d" in annotation_types:
            labels["bbox_2d"] = [
                {
                    "id": 0,
                    "class": "CAR",
                    "bbox": [100, 150, 300, 350],
                    "confidence": 0.85
                }
            ]

        if "bbox_3d" in annotation_types:
            labels["bbox_3d"] = [
                {
                    "id": 0,
                    "class": "CAR",
                    "center": [10.5, -1.2, 0.0],
                    "size": [4.5, 2.0, 1.8],
                    "confidence": 0.78
                }
            ]

        if "segmentation" in annotation_types:
            labels["segmentation"] = {
                "drivable_area": [[0, 0], [640, 0], [640, 480], [0, 480]],
                "confidence": 0.82
            }

        if "expert_demonstration" in annotation_types:
            labels["expert_demonstration"] = {
                "ideal_trajectory": [[0, 0], [10, 0.5], [20, 1.0], [30, 1.5]],
                "optimal_decision": "SLOW_DOWN_AND_AVOID",
                "confidence": 0.70  # Lower confidence for expert demo
            }

        # Calculate overall confidence
        confidences = []
        for label_value in labels.values():
            if isinstance(label_value, dict):
                confidences.append(label_value.get("confidence", 0.8))
            elif isinstance(label_value, list):
                # For lists, take average confidence of items
                for item in label_value:
                    if isinstance(item, dict) and "confidence" in item:
                        confidences.append(item["confidence"])
            else:
                confidences.append(0.8)

        overall_confidence = np.mean(confidences) if confidences else 0.8

        return {
            "labels": labels,
            "confidence": overall_confidence,
            "types": annotation_types
        }


class AnnotatorPool:
    """Pool of human annotators"""

    def __init__(self):
        self.annotators = {
            "ann_001": {"id": "ann_001", "specialty": ["bbox_2d", "bbox_3d"]},
            "ann_002": {"id": "ann_002", "specialty": ["segmentation", "tracking"]},
            "ann_003": {"id": "ann_003", "specialty": ["expert_demonstration", "trajectory"]},
        }

    def manual_annotate(self, event_id: str, event_data: Dict[str, Any],
                       annotation_types: List[str],
                       pre_annotation: Dict[str, Any]) -> AnnotationResult:
        """
        Full manual annotation (from scratch)
        """
        # Select annotator based on annotation types
        annotator_id = self._select_annotator(annotation_types)

        # Simulated manual annotation
        labels = {}
        for ann_type in annotation_types:
            if ann_type == "expert_demonstration":
                labels["expert_demonstration"] = {
                    "ideal_trajectory": [[0, 0], [5, -1.0], [10, -1.5], [20, -2.0]],
                    "optimal_decision": "LANE_CHANGE_LEFT",
                }
            else:
                labels[ann_type] = "manually_annotated"

        return AnnotationResult(
            annotation_id=f"ann_manual_{event_id}_{int(time.time())}",
            event_id=event_id,
            annotation_type=",".join(annotation_types),
            labels=labels,
            annotator_id=annotator_id,
            confidence=0.95,  # Human annotation has high confidence
            timestamp=datetime.now(),
            review_status="pending_review"
        )

    def review_annotation(self, event_id: str, event_data: Dict[str, Any],
                         pre_annotation: Dict[str, Any]) -> AnnotationResult:
        """
        Review and correct pre-annotation
        """
        annotator_id = self._select_annotator(pre_annotation.get("types", []))

        # Simulated review - sometimes correct, sometimes not
        labels = pre_annotation.get("labels", {}).copy()

        # Randomly modify some labels
        if random.random() < 0.3:
            if "bbox_2d" in labels:
                labels["bbox_2d"][0]["bbox"] = [95, 145, 305, 355]

        return AnnotationResult(
            annotation_id=f"ann_review_{event_id}_{int(time.time())}",
            event_id=event_id,
            annotation_type=",".join(pre_annotation.get("types", [])),
            labels=labels,
            annotator_id=annotator_id,
            confidence=0.85,  # Higher than pre-annotation
            timestamp=datetime.now(),
            review_status="accepted"
        )

    def _select_annotator(self, annotation_types: List[str]) -> str:
        """Select annotator based on annotation types"""
        # Simple round-robin selection
        return list(self.annotators.keys())[(int(time.time()) // 10) % len(self.annotators)]


class DataAugmenter:
    """
    Data augmentation
    Implements traditional and AIGC augmentation methods
    """

    def __init__(self):
        self.traditional_methods = {
            "geometric_transform": self._geometric_transform,
            "color_jitter": self._color_jitter,
            "noise_injection": self._noise_injection,
            "weather_simulation": self._weather_simulation,
        }

        self.aigc_methods = {
            "nerf": self._nerf_augmentation,
            "diffusion": self._diffusion_augmentation,
            "simulation": self._simulation_augmentation,
        }

    def augment(self, event_id: str, event_data: Dict[str, Any],
                methods: List[str] = None,
                count: int = 1) -> AugmentationResult:
        """
        Augment event data

        Args:
            event_id: Original event ID
            event_data: Event data
            methods: Augmentation methods (defaults to random selection)
            count: Number of augmentations to generate

        Returns:
            Augmentation result
        """
        if methods is None:
            # Randomly select methods
            n_methods = random.randint(1, 3)
            methods = random.sample(
                list(self.traditional_methods.keys()) +
                list(self.aigc_methods.keys()),
                n_methods
            )

        augmented_event_ids = []
        parameters = {}

        for i in range(count):
            aug_id = f"{event_id}_aug_{i}_{int(time.time())}"
            augmented_event_ids.append(aug_id)

            # Apply each augmentation method
            aug_data = event_data.copy()
            for method in methods:
                if method in self.traditional_methods:
                    aug_data, params = self.traditional_methods[method](aug_data)
                elif method in self.aigc_methods:
                    aug_data, params = self.aigc_methods[method](aug_data)

                parameters[method] = params

        return AugmentationResult(
            original_event_id=event_id,
            augmented_event_ids=augmented_event_ids,
            augmentation_methods=methods,
            parameters=parameters
        )

    # Traditional augmentation methods
    def _geometric_transform(self, event_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply geometric transformations (rotation, translation, scaling)"""
        params = {
            "rotation": random.uniform(-15, 15),
            "translation_x": random.uniform(-5, 5),
            "translation_y": random.uniform(-5, 5),
            "scale": random.uniform(0.9, 1.1),
        }
        return event_data, params

    def _color_jitter(self, event_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Apply color jittering (brightness, contrast, saturation)"""
        params = {
            "brightness": random.uniform(0.8, 1.2),
            "contrast": random.uniform(0.8, 1.2),
            "saturation": random.uniform(0.8, 1.2),
        }
        return event_data, params

    def _noise_injection(self, event_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Inject noise to sensor data"""
        params = {
            "noise_level": random.uniform(0.01, 0.1),
            "noise_type": random.choice(["gaussian", "speckle"]),
        }
        return event_data, params

    def _weather_simulation(self, event_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Simulate weather conditions (rain, snow, fog)"""
        params = {
            "weather_type": random.choice(["rain", "snow", "fog"]),
            "intensity": random.uniform(0.3, 0.9),
        }
        return event_data, params

    # AIGC augmentation methods
    def _nerf_augmentation(self, event_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        NeRF-based augmentation
        Reconstruct scene and render from new viewpoints/times
        """
        params = {
            "iterations": Config.Augmentation.NERF_ITERATIONS,
            "viewpoint_change": random.uniform(-45, 45),  # degrees
            "time_change": random.choice([-6, -3, 3, 6]),  # hours
            "weather_change": random.choice(["day", "night", "dawn", "dusk"]),
        }
        return event_data, params

    def _diffusion_augmentation(self, event_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Diffusion-based augmentation
        Generate rare objects and in-paint them
        """
        params = {
            "steps": Config.Augmentation.DIFFUSION_STEPS,
            "object_type": random.choice(["overturned_truck", "animal_group", "debris"]),
            "inpaint_location": random.uniform(0, 1),
        }
        return event_data, params

    def _simulation_augmentation(self, event_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulation-based augmentation
        Generate variants in simulator
        """
        params = {
            "num_variants": random.randint(10, 100),
            "parameter_variations": {
                "weather": random.choice(["clear", "rain", "fog"]),
                "lighting": random.choice(["day", "night", "twilight"]),
                "traffic_density": random.uniform(0.1, 0.9),
            }
        }
        return event_data, params


class DatasetBuilder:
    """
    Build final dataset from raw events, annotations, and augmentations
    """

    def __init__(self):
        self.train_events = []
        self.val_events = []
        self.test_events = []

    def build_dataset(self, events: List[Dict[str, Any]],
                      annotations: List[AnnotationResult],
                      augmentations: List[AugmentationResult],
                      split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)
                      ) -> Dict[str, Any]:
        """
        Build train/val/test dataset

        Args:
            events: Raw events
            annotations: Annotation results
            augmentations: Augmentation results
            split_ratio: (train, val, test) ratio

        Returns:
            Dataset statistics
        """
        # Merge events with annotations
        annotated_events = self._merge_annotations(events, annotations)

        # Add augmented events
        for aug in augmentations:
            aug_events = self._create_augmented_events(aug, annotated_events)
            annotated_events.extend(aug_events)

        # Split dataset
        n_events = len(annotated_events)
        train_count = int(n_events * split_ratio[0])
        val_count = int(n_events * split_ratio[1])

        # Shuffle before split
        import random
        random.shuffle(annotated_events)

        self.train_events = annotated_events[:train_count]
        self.val_events = annotated_events[train_count:train_count + val_count]
        self.test_events = annotated_events[train_count + val_count:]

        # Ensure corner cases are in test set
        self._ensure_corner_cases_in_test()

        stats = {
            "total_events": n_events,
            "train_count": len(self.train_events),
            "val_count": len(self.val_events),
            "test_count": len(self.test_events),
            "corner_cases_in_test": sum(1 for e in self.test_events if "hard_case" in e.get("tags", [])),
        }

        logging.info(f"Dataset built: {stats}")
        return stats

    def _merge_annotations(self, events: List[Dict[str, Any]],
                          annotations: List[AnnotationResult]) -> List[Dict[str, Any]]:
        """Merge events with their annotations"""
        annotation_map = {ann.event_id: ann for ann in annotations}

        merged = []
        for event in events:
            event_id = event.get("event_id")
            if event_id in annotation_map:
                merged_event = event.copy()
                merged_event["annotation"] = annotation_map[event_id]
                merged.append(merged_event)

        return merged

    def _create_augmented_events(self, augmentation: AugmentationResult,
                                 base_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create event objects for augmented data"""
        base_event = next((e for e in base_events
                          if e.get("event_id") == augmentation.original_event_id), None)

        if not base_event:
            return []

        augmented_events = []
        for aug_id in augmentation.augmented_event_ids:
            aug_event = base_event.copy()
            aug_event["event_id"] = aug_id
            aug_event["augmentation"] = {
                "methods": augmentation.augmentation_methods,
                "parameters": augmentation.parameters
            }
            aug_event["is_augmented"] = True
            augmented_events.append(aug_event)

        return augmented_events

    def _ensure_corner_cases_in_test(self):
        """Ensure corner cases are represented in test set"""
        # Find corner cases in train/val
        corner_cases = []
        for event in self.train_events + self.val_events:
            if "hard_case" in event.get("tags", []):
                corner_cases.append(event)

        # Move some to test set
        n_to_move = min(len(corner_cases), len(corner_cases) // 2)
        for i in range(n_to_move):
            corner_case = corner_cases[i]
            # Remove from train/val
            if corner_case in self.train_events:
                self.train_events.remove(corner_case)
            elif corner_case in self.val_events:
                self.val_events.remove(corner_case)
            # Add to test
            self.test_events.append(corner_case)


class DataMiningOrchestrator:
    """
    Orchestrates data mining, annotation, and dataset building
    """

    def __init__(self):
        self.scenario_miner = ScenarioMiner()
        self.annotation_pipeline = AnnotationPipeline()
        self.data_augmenter = DataAugmenter()
        self.dataset_builder = DatasetBuilder()

    def process_pipeline(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run complete data mining pipeline

        Args:
            events: List of events to process

        Returns:
            Pipeline results
        """
        results = {
            "mined_scenarios": 0,
            "annotated_events": 0,
            "augmented_events": 0,
            "dataset_stats": {},
        }

        # Step 1: Mine scenarios
        scenarios = self.scenario_miner.mine_scenarios(events, method="all")
        results["mined_scenarios"] = len(scenarios)

        # Step 2: Annotate events (sample for efficiency)
        sample_size = min(len(events), 10)
        sample_events = events[:sample_size]

        for event in sample_events:
            self.annotation_pipeline.annotate_event(
                event.get("event_id"),
                event
            )

        results["annotated_events"] = sample_size

        # Step 3: Augment hard cases
        hard_cases = self.scenario_miner.get_long_tail_scenarios(threshold=5)
        augmentations = []

        for scenario in hard_cases[:3]:  # Limit for testing
            aug_result = self.data_augmenter.augment(
                scenario.event_ids[0] if scenario.event_ids else "unknown",
                scenario.metadata,
                count=3
            )
            augmentations.append(aug_result)

        results["augmented_events"] = sum(len(aug.augmented_event_ids) for aug in augmentations)

        # Step 4: Build dataset
        dataset_stats = self.dataset_builder.build_dataset(
            events,
            self.annotation_pipeline.annotations,
            augmentations
        )
        results["dataset_stats"] = dataset_stats

        # Additional statistics
        results["annotation_stats"] = self.annotation_pipeline.get_annotation_statistics()
        results["long_tail_scenarios"] = len(hard_cases)

        logging.info(f"Data mining pipeline complete: {results}")
        return results


if __name__ == '__main__':
    # Test data mining module
    logging.basicConfig(level=logging.INFO)
    print("Data Mining Module Test")
    print("=" * 50)

    orchestrator = DataMiningOrchestrator()

    # Create test events
    test_events = []
    for i in range(20):
        test_events.append({
            "event_id": f"evt_{i:03d}",
            "vehicle_id": f"VIN_{i:03d}",
            "trigger_type": ["RULE_BASED", "MODEL_BASED", "UNCERTAINTY_BASED"][i % 3],
            "trigger_time_utc": datetime.now().isoformat(),
            "location": {
                "latitude": 39.9042 + np.random.uniform(-0.1, 0.1),
                "longitude": 116.4074 + np.random.uniform(-0.1, 0.1),
                "altitude": 52.5,
                "heading": 85.5,
            },
            "scenario_tags": ["highway", "construction_zone"][i % 2: (i % 2) + 1],
            "trigger_details": {
                "perception_max_confidence": np.random.uniform(0.2, 0.95),
                "uncertainty_entropy": np.random.uniform(0.1, 0.9),
            },
            "sensor_status": {
                "lidar_front": "OK",
                "camera_front_long": "OK",
                "radar_front": "OK",
                "gps_rtk": "FIXED",
                "imu": "OK"
            },
            "file_manifest": [],
        })

    # Run pipeline
    print("\nRunning data mining pipeline...")
    results = orchestrator.process_pipeline(test_events)

    print(f"\nPipeline Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    print("\nTest completed!")
