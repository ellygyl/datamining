#!/usr/bin/env python3
"""
AutopilotDataLoopOrchestrator - 自动驾驶数据闭环系统
实现车端异常触发 -> 云端挖掘 -> 模型训练的自动化 Pipeline
"""

import json
import re
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import random
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """触发类型枚举"""
    RULE_BASED = "rule_based"
    MODEL_BASED = "model_based"
    SHADOW_MODE = "shadow_mode"
    UNCERTAINTY = "uncertainty"


class DataType(Enum):
    """数据类型枚举"""
    SMALL = "small"  # < 10 KB
    MEDIUM = "medium"  # < 100 MB
    LARGE = "large"  # >= 100 MB


@dataclass
class CANSignal:
    """CAN 总线信号"""
    longitudinal_acc: float  # 纵向加速度 m/s²
    lateral_acc: float  # 横向加速度 m/s²
    speed: float  # 车速 km/h
    speed_delta_1s: float  # 1秒内速度差 km/h
    brake_pedal: float  # 制动踏板开度 %
    steering_wheel_velocity: float  # 方向盘角速度 °/s
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerceptionResult:
    """感知结果"""
    object_confidence: float  # 目标检测最大置信度
    ood_score: float  # 分布外检测分数
    trajectory_error: float  # 轨迹预测误差
    entropy: float  # 输出熵（不确定性）
    detected_objects: List[str] = field(default_factory=list)


@dataclass
class VehicleEvent:
    """车辆事件"""
    event_id: str
    vehicle_id: str
    trigger_type: TriggerType
    trigger_reason: str
    can_signal: CANSignal
    perception_result: Optional[PerceptionResult] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class VehicleTriggerStrategy:
    """车端触发策略 - 三级漏斗机制"""

    # 阈值配置
    RULE_LONG_ACC_THRESHOLD = -4.0  # m/s²
    RULE_LAT_ACC_THRESHOLD = 3.0  # m/s²
    RULE_SPEED_THRESHOLD = 10.0  # km/h
    RULE_SPEED_DELTA_1S_LOW = 10.0  # km/h
    RULE_SPEED_DELTA_1S_HIGH = 30.0  # km/h
    RULE_BRAKE_LOW = 20.0  # %
    RULE_BRAKE_HIGH = 40.0  # %
    RULE_BRAKE_AEB = 80.0  # %

    MODEL_CONF_THRESHOLD = 0.4  # 置信度阈值
    MODEL_OOD_THRESHOLD = 0.7  # OOD分数阈值
    MODEL_TRAJ_ERROR_THRESHOLD = 2.0  # 轨迹误差阈值

    SHADOW_ENTROPY_THRESHOLD = 1.5  # 熵阈值

    def __init__(self):
        self.production_model = None  # 量产模型
        self.shadow_model = None  # 影子模型

    def check_rule_based_trigger(self, can: CANSignal) -> Optional[str]:
        """
        第一层：硬规则触发
        """
        reasons = []

        # 急刹检测
        if (can.longitudinal_acc < self.RULE_LONG_ACC_THRESHOLD and
            can.speed > self.RULE_SPEED_THRESHOLD):
            reasons.append("急刹")

        # 急拐检测
        if (can.lateral_acc > self.RULE_LAT_ACC_THRESHOLD and
            can.speed > self.RULE_SPEED_THRESHOLD):
            reasons.append("急拐")

        # 速度差 + 制动踏板检测（低档）
        if (can.speed_delta_1s > self.RULE_SPEED_DELTA_1S_LOW and
            can.brake_pedal > self.RULE_BRAKE_LOW):
            reasons.append(f"急减速_速度差{can.speed_delta_1s:.1f}")

        # 速度差 + 制动踏板检测（高档）
        if (can.speed_delta_1s > self.RULE_SPEED_DELTA_1S_HIGH and
            can.brake_pedal > self.RULE_BRAKE_HIGH):
            reasons.append("AEB介入前兆_高档")

        # 方向盘角速度突变
        if abs(can.steering_wheel_velocity) > 100:
            reasons.append("方向盘突变")

        # 制动踏板 AEB 阈值
        if can.brake_pedal > self.RULE_BRAKE_AEB:
            reasons.append("制动踏板AEB")

        return ", ".join(reasons) if reasons else None

    def check_model_based_trigger(self, perception: PerceptionResult) -> Optional[str]:
        """
        第二层：轻量级感知异常检测
        """
        reasons = []

        # 目标检测置信度低
        if perception.object_confidence < self.MODEL_CONF_THRESHOLD:
            reasons.append(f"检测置信度低({perception.object_confidence:.2f})")

        # 分布外检测
        if perception.ood_score > self.MODEL_OOD_THRESHOLD:
            reasons.append(f"OOD检测({perception.ood_score:.2f})")

        # 轨迹预测偏差
        if perception.trajectory_error > self.MODEL_TRAJ_ERROR_THRESHOLD:
            reasons.append(f"轨迹误差高({perception.trajectory_error:.2f})")

        return ", ".join(reasons) if reasons else None

    def check_shadow_mode_trigger(self, perception: PerceptionResult) -> Optional[str]:
        """
        第三层：影子模式与不确定性估算
        """
        reasons = []

        # 不确定性量化
        if perception.entropy > self.SHADOW_ENTROPY_THRESHOLD:
            reasons.append(f"高不确定性(熵={perception.entropy:.2f})")

        # 影子模式：模拟决策分歧
        # 这里简化为随机模拟
        if random.random() < 0.05:  # 5%概率模拟决策分歧
            reasons.append("影子模式决策分歧")

        return ", ".join(reasons) if reasons else None

    def evaluate_trigger(self, can: CANSignal, perception: Optional[PerceptionResult] = None) -> Optional[VehicleEvent]:
        """
        评估是否触发事件
        """
        vehicle_id = f"VEH_{random.randint(10000, 99999)}"
        event_id = f"EVT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"

        # 第一层：硬规则
        rule_reason = self.check_rule_based_trigger(can)
        if rule_reason:
            return VehicleEvent(
                event_id=event_id,
                vehicle_id=vehicle_id,
                trigger_type=TriggerType.RULE_BASED,
                trigger_reason=rule_reason,
                can_signal=can,
                perception_result=perception
            )

        # 第二、三层：感知模型
        if perception:
            model_reason = self.check_model_based_trigger(perception)
            if model_reason:
                return VehicleEvent(
                    event_id=event_id,
                    vehicle_id=vehicle_id,
                    trigger_type=TriggerType.MODEL_BASED,
                    trigger_reason=model_reason,
                    can_signal=can,
                    perception_result=perception
                )

            shadow_reason = self.check_shadow_mode_trigger(perception)
            if shadow_reason:
                return VehicleEvent(
                    event_id=event_id,
                    vehicle_id=vehicle_id,
                    trigger_type=TriggerType.SHADOW_MODE,
                    trigger_reason=shadow_reason,
                    can_signal=can,
                    perception_result=perception
                )

        return None


class DataTransmissionManager:
    """数据传输管理器"""

    THRESHOLD_SMALL = 10 * 1024  # 10 KB
    THRESHOLD_LARGE = 100 * 1024 * 1024  # 100 MB

    def __init__(self):
        self.upload_queue = []

    def determine_data_type(self, size_bytes: int) -> DataType:
        """确定数据类型"""
        if size_bytes < self.THRESHOLD_SMALL:
            return DataType.SMALL
        elif size_bytes < self.THRESHOLD_LARGE:
            return DataType.MEDIUM
        else:
            return DataType.LARGE

    def upload_small_data(self, metadata: Dict) -> bool:
        """小包上传 - MQTT over TLS"""
        logger.info(f"MQTT上传小包: {metadata.get('event_id')}")
        return True

    def upload_medium_data(self, event: VehicleEvent, data_bytes: bytes) -> bool:
        """中包上传 - gRPC/HTTP2"""
        logger.info(f"gRPC上传中包: {event.event_id}, 大小: {len(data_bytes) / 1024 / 1024:.2f} MB")
        return True

    def upload_large_data(self, event: VehicleEvent, data_bytes: bytes) -> bool:
        """大包上传 - QUIC"""
        logger.info(f"QUIC上传大包: {event.event_id}, 大小: {len(data_bytes) / 1024 / 1024:.2f} MB")
        return True

    def upload_event_data(self, event: VehicleEvent, data_size: int) -> bool:
        """根据数据类型选择上传方式"""
        data_type = self.determine_data_type(data_size)

        metadata = {
            "event_id": event.event_id,
            "vehicle_id": event.vehicle_id,
            "trigger_type": event.trigger_type.value,
            "trigger_reason": event.trigger_reason,
            "timestamp": event.timestamp.isoformat(),
            "can_data": {
                "longitudinal_acc": event.can_signal.longitudinal_acc,
                "lateral_acc": event.can_signal.lateral_acc,
                "speed": event.can_signal.speed
            }
        }

        # 上传元数据
        self.upload_small_data(metadata)

        # 上传主数据
        if data_type == DataType.SMALL:
            return True
        elif data_type == DataType.MEDIUM:
            return self.upload_medium_data(event, b"x" * data_size)
        else:
            return self.upload_large_data(event, b"x" * data_size)


class CloudGateway:
    """云端网关 - 认证与路由"""

    def __init__(self):
        self.valid_dids = set()

    def verify_did(self, vehicle_id: str, did_cert: str) -> bool:
        """验证车辆 DID 证书"""
        # 简化验证逻辑
        return did_cert.startswith("DID_") and vehicle_id.startswith("VEH_")

    def route_event(self, event: Dict) -> str:
        """路由事件到对应的消息队列"""
        priority = self._determine_priority(event)
        return f"topic_{priority}"

    def _determine_priority(self, event: Dict) -> str:
        """确定事件优先级"""
        trigger_type = event.get("trigger_type", "")
        if "rule_based" in trigger_type:
            return "high_priority"
        elif "shadow_mode" in trigger_type:
            return "medium_priority"
        else:
            return "low_priority"


class CloudStorageManager:
    """云端存储管理器 - 分级存储"""

    def __init__(self):
        self.oss_bucket = "autopilot_raw_data"
        self.iotdb_db = "can_timeseries"
        self.pg_index = "metadata_index"

    def store_unstructured_data(self, event_id: str, vehicle_id: str,
                                  data_type: str, data: bytes) -> str:
        """存储非结构化数据到对象存储"""
        date = datetime.now().strftime("%Y%m%d")
        path = f"{self.oss_bucket}/{date}/{vehicle_id}/{event_id}/{data_type}.bin"
        logger.info(f"OSS存储: {path}")
        return path

    def store_timeseries_data(self, vehicle_id: str, can_data: Dict) -> bool:
        """存储时序数据到 IoTDB"""
        logger.info(f"IoTDB存储: 车辆{vehicle_id} CAN数据")
        return True

    def store_metadata(self, event: Dict) -> bool:
        """存储元数据到 PostgreSQL"""
        logger.info(f"PG存储: 事件{event.get('event_id')} 元数据")
        return True

    def store_graph_data(self, vehicle_id: str, location: Dict) -> bool:
        """存储图数据到 Neo4j"""
        logger.info(f"Neo4j存储: 车辆{vehicle_id} 位置信息")
        return True


class StreamProcessor:
    """流处理器 - Flink 实时处理"""

    def __init__(self):
        self.alerts = []
        self.vehicle_health = {}

    def consume_kafka_message(self, message: Dict) -> Optional[Dict]:
        """消费 Kafka 消息"""
        event = json.loads(message) if isinstance(message, str) else message

        # 实时解析元数据
        self._update_vehicle_health(event)

        # 实时告警检查
        alert = self._check_high_risk_scenario(event)
        if alert:
            self._send_alert(alert)

        return event

    def _update_vehicle_health(self, event: Dict):
        """更新车辆健康状态"""
        vehicle_id = event.get("vehicle_id")
        trigger_type = event.get("trigger_type")

        if vehicle_id not in self.vehicle_health:
            self.vehicle_health[vehicle_id] = {"events": 0, "last_update": datetime.now()}

        self.vehicle_health[vehicle_id]["events"] += 1
        self.vehicle_health[vehicle_id]["last_update"] = datetime.now()

    def _check_high_risk_scenario(self, event: Dict) -> Optional[Dict]:
        """检查高危场景"""
        trigger_reason = event.get("trigger_reason", "")

        # AEB 频繁触发
        if "AEB" in trigger_reason:
            return {
                "vehicle_id": event.get("vehicle_id"),
                "alert_type": "HIGH_RISK",
                "message": "检测到AEB频繁触发",
                "timestamp": datetime.now().isoformat()
            }
        return None

    def _send_alert(self, alert: Dict):
        """发送告警"""
        self.alerts.append(alert)
        logger.warning(f"告警: {alert}")


class BatchProcessor:
    """批处理器 - Spark 离线处理"""

    def __init__(self):
        self.processed_count = 0

    def run_etl_job(self, date_str: str) -> int:
        """运行 ETL 作业"""
        logger.info(f"启动 Spark ETL 作业: {date_str}")

        # 数据清洗
        cleaned = self._clean_data(date_str)

        # 去重
        deduplicated = self._deduplicate(cleaned)

        # 时间同步对齐
        aligned = self._time_sync(deduplicated)

        self.processed_count = len(aligned)
        logger.info(f"ETL 完成，处理 {self.processed_count} 条记录")

        return self.processed_count

    def _clean_data(self, date_str: str) -> List[Dict]:
        """数据清洗"""
        # 剔除传感器故障、时间戳不同步、画面模糊数据
        logger.info("数据清洗中...")
        return [{"event_id": f"EVT_{i}", "status": "cleaned"} for i in range(100)]

    def _deduplicate(self, data: List[Dict]) -> List[Dict]:
        """去重"""
        logger.info("数据去重中...")
        return data[:80]  # 简化去重逻辑

    def _time_sync(self, data: List[Dict]) -> List[Dict]:
        """多传感器时间同步对齐"""
        logger.info("时间同步对齐中...")
        return data[:70]  # 简化同步逻辑


class DataMiner:
    """数据挖掘器"""

    def __init__(self):
        self.scenario_library = []

    def mine_scenarios(self, filters: Dict) -> List[Dict]:
        """挖掘场景"""
        logger.info(f"根据规则挖掘场景: {filters}")

        # 规则过滤
        filtered = self._filter_by_rules(filters)

        # 聚类分析
        clustered = self._cluster_scenarios(filtered)

        # 难例挖掘
        hard_cases = self._mine_hard_cases(filtered)

        return clustered + hard_cases

    def _filter_by_rules(self, filters: Dict) -> List[Dict]:
        """规则过滤"""
        weather = filters.get("weather")
        object_type = filters.get("object_type")

        results = []
        for i in range(50):
            scenario = {
                "scenario_id": f"S_{i}",
                "weather": weather or random.choice(["rain", "clear", "fog"]),
                "object_type": object_type or random.choice(["car", "pedestrian", "construction_zone"]),
                "difficulty": random.random()
            }
            if (weather is None or scenario["weather"] == weather) and \
               (object_type is None or scenario["object_type"] == object_type):
                results.append(scenario)

        return results

    def _cluster_scenarios(self, scenarios: List[Dict]) -> List[Dict]:
        """聚类分析 - K-Means/DBSCAN"""
        logger.info("场景聚类分析中...")
        # 模拟聚类结果
        for i, s in enumerate(scenarios):
            s["cluster"] = i % 5
        return scenarios

    def _mine_hard_cases(self, scenarios: List[Dict]) -> List[Dict]:
        """难例挖掘"""
        logger.info("难例挖掘中...")
        # 筛选 Loss 最高的样本
        hard_cases = sorted(scenarios, key=lambda x: x.get("difficulty", 0), reverse=True)[:10]
        for h in hard_cases:
            h["is_hard_case"] = True
        return hard_cases


class AnnotationPipeline:
    """人机协同标注流水线"""

    CONFIDENCE_HIGH = 0.9
    CONFIDENCE_LOW = 0.6

    def __init__(self):
        self.pending_tasks = []
        self.completed_annotations = []

    def pre_annotation(self, event: VehicleEvent) -> Dict:
        """预标注 - 使用大模型自动标注"""
        logger.info(f"预标注: {event.event_id}")

        # 模拟大模型预标注结果
        confidence = random.uniform(0.5, 0.95)

        annotation = {
            "event_id": event.event_id,
            "2d_boxes": [
                {"class": "car", "bbox": [100, 100, 200, 200], "conf": confidence},
                {"class": "barrier", "bbox": [300, 150, 400, 250], "conf": confidence - 0.1}
            ],
            "3d_boxes": [],
            "semantic_segmentation": [],
            "future_trajectory": [],
            "confidence": confidence
        }

        return annotation

    def route_annotation_task(self, annotation: Dict) -> str:
        """根据置信度分流标注任务"""
        conf = annotation.get("confidence", 0)

        if conf > self.CONFIDENCE_HIGH:
            # 直接入库（免检）
            return "direct_accept"
        elif conf > self.CONFIDENCE_LOW:
            # 人工抽检
            return "spot_check"
        else:
            # 人工全量复核
            return "full_review"

    def human_correction(self, annotation: Dict, correction_type: str) -> Dict:
        """人工修正"""
        logger.info(f"人工修正: {correction_type}")

        # 模拟人工修正
        corrected = annotation.copy()
        corrected["human_verified"] = True
        corrected["correction_type"] = correction_type

        return corrected

    def expert_demonstration(self, event: VehicleEvent) -> Dict:
        """专家重演 - 针对接管场景"""
        logger.info(f"专家重演: {event.event_id}")

        demo = {
            "event_id": event.event_id,
            "ideal_trajectory": [[0, 0], [10, 0], [20, 1], [30, 3]],
            "optimal_decision": "lane_change_right",
            "expert_id": f"EXP_{random.randint(1, 100)}"
        }

        return demo


class AugmentationEngine:
    """高级数据增强引擎"""

    def traditional_augmentation(self, data: Dict) -> List[Dict]:
        """传统数据增强"""
        logger.info("传统数据增强...")

        augmented = []
        augmentations = ["rotate", "scale", "brightness", "noise", "weather"]

        for aug in augmentations:
            new_data = data.copy()
            new_data["augmentation"] = aug
            augmented.append(new_data)

        return augmented

    def aigc_augmentation(self, data: Dict, method: str) -> Dict:
        """生成式增强 (AIGC)"""
        logger.info(f"AIGC增强: {method}")

        if method == "nerf":
            # NeRF 重建场景
            return {
                **data,
                "augmentation": "nerf",
                "viewpoint": "changed",
                "time": "night",
                "weather": "rain"
            }
        elif method == "diffusion":
            # Diffusion 生成稀有障碍物
            return {
                **data,
                "augmentation": "diffusion_inpainting",
                "generated_objects": ["overturned_car", "animals"]
            }

        return data

    def simulation_generalization(self, scenario: Dict, num_variants: int = 1000) -> List[Dict]:
        """仿真泛化 - 生成大量变体"""
        logger.info(f"仿真泛化: 生成 {num_variants} 个场景变体")

        variants = []
        for i in range(num_variants):
            variant = scenario.copy()
            variant["variant_id"] = f"VAR_{i}"
            variant["parameters"] = {
                "time_of_day": random.choice(["day", "night", "dawn", "dusk"]),
                "weather": random.choice(["clear", "rain", "snow", "fog"]),
                "traffic_density": random.randint(0, 100),
                "lighting": random.uniform(0, 100)
            }
            variants.append(variant)

        return variants


class ModelTrainingPipeline:
    """模型训练流水线"""

    def __init__(self):
        self.model_registry = {}

    def build_dataset(self, raw_data: List, annotations: List, augmented: List) -> Dict:
        """构建数据集"""
        logger.info("构建数据集...")

        total_size = len(raw_data) + len(annotations) + len(augmented)

        # 数据集划分
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.15)
        test_size = total_size - train_size - val_size

        dataset = {
            "train": train_size,
            "val": val_size,
            "test": test_size,
            "total": total_size
        }

        logger.info(f"数据集划分: Train={train_size}, Val={val_size}, Test={test_size}")

        return dataset

    def train_model(self, dataset: Dict, model_type: str) -> Dict:
        """模型训练"""
        logger.info(f"开始训练 {model_type} 模型...")

        # 加载预训练权重
        weights = "pretrained_weights.pth"
        logger.info(f"加载预训练权重: {weights}")

        # 分布式训练
        logger.info("启动分布式训练 (PyTorch DDP/FSDP)...")

        # 模拟训练过程
        epochs = 100
        for epoch in range(1, epochs + 1):
            loss = random.uniform(0.1, 1.0)
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

            if epoch % 10 == 0:
                logger.info(f"保存检查点: checkpoint_epoch_{epoch}.pth")

        logger.info("训练完成")

        return {
            "model_type": model_type,
            "epochs": epochs,
            "final_loss": 0.05,
            "checkpoint": f"{model_type}_final.pth"
        }

    def offline_evaluation(self, model: Dict, test_dataset: List) -> Dict:
        """离线评估"""
        logger.info("离线评估中...")

        # 计算 mAP, Recall, ADE/FDE 等指标
        metrics = {
            "mAP": random.uniform(0.7, 0.95),
            "Recall": random.uniform(0.8, 0.98),
            "ADE": random.uniform(0.5, 2.0),
            "FDE": random.uniform(1.0, 3.0)
        }

        logger.info(f"评估指标: {metrics}")

        return metrics

    def closed_loop_simulation(self, model: Dict, scenarios: List) -> Dict:
        """闭环仿真"""
        logger.info("闭环仿真测试...")

        # 回归测试
        regression_passed = self._run_regression_test(model, scenarios)
        logger.info(f"回归测试通过率: {regression_passed:.2%}")

        # 压力测试
        stress_passed = self._run_stress_test(model, 100000)
        logger.info(f"压力测试通过率: {stress_passed:.2%}")

        return {
            "regression_pass_rate": regression_passed,
            "stress_pass_rate": stress_passed,
            "overall_pass": regression_passed > 0.95 and stress_passed > 0.9
        }

    def _run_regression_test(self, model: Dict, scenarios: List) -> float:
        """回归测试 - 重跑历史失败案例"""
        passed = sum(1 for _ in scenarios if random.random() > 0.05)
        return passed / len(scenarios) if scenarios else 0

    def _run_stress_test(self, model: Dict, num_scenarios: int) -> float:
        """压力测试 - 百万级随机场景"""
        passed = int(num_scenarios * 0.98)
        return passed / num_scenarios

    def generate_ota_package(self, model: Dict, version: str) -> str:
        """生成 OTA 包"""
        logger.info(f"生成 OTA 包: v{version}")

        ota_package = {
            "version": version,
            "model_checkpoint": model.get("checkpoint"),
            "compatible_vehicles": ["TRUCK_MODEL_X", "TRUCK_MODEL_Y"],
            "deployment_strategy": "canary",  # 灰度发布
            "canary_ratio": 0.1  # 10% 灰度
        }

        logger.info(f"OTA 包已生成: {ota_package}")

        return json.dumps(ota_package)


class AutopilotDataLoopOrchestrator:
    """
    自动驾驶数据闭环编排器 - 主类
    协调车端触发、云端挖掘、模型训练全流程
    """

    def __init__(self):
        # 车端组件
        self.trigger_strategy = VehicleTriggerStrategy()
        self.data_transmission = DataTransmissionManager()

        # 云端组件
        self.gateway = CloudGateway()
        self.storage = CloudStorageManager()
        self.stream_processor = StreamProcessor()
        self.batch_processor = BatchProcessor()

        # 数据处理组件
        self.data_miner = DataMiner()
        self.annotation_pipeline = AnnotationPipeline()
        self.augmentation_engine = AugmentationEngine()

        # 训练组件
        self.training_pipeline = ModelTrainingPipeline()

    def process_vehicle_event(self, can: CANSignal, perception: Optional[PerceptionResult] = None) -> Dict:
        """
        处理车辆事件 - 完整流程
        """
        logger.info("=" * 50)
        logger.info("开始处理车辆事件")

        # 1. 车端触发评估
        event = self.trigger_strategy.evaluate_trigger(can, perception)
        if not event:
            logger.info("未触发事件")
            return {"status": "no_trigger"}

        logger.info(f"事件触发: {event.event_id}, 类型: {event.trigger_type.value}, 原因: {event.trigger_reason}")

        # 2. 数据上传
        data_size = random.randint(1000, 50 * 1024 * 1024)  # 模拟数据大小
        upload_success = self.data_transmission.upload_event_data(event, data_size)
        if not upload_success:
            logger.error("数据上传失败")
            return {"status": "upload_failed"}

        # 3. 云端接入
        metadata = self._create_metadata(event)
        topic = self.gateway.route_event(metadata)
        logger.info(f"事件路由到: {topic}")

        # 4. 存储数据
        self._store_event_data(event, metadata, data_size)

        # 5. 实时流处理
        self.stream_processor.consume_kafka_message(metadata)

        return {
            "status": "processed",
            "event_id": event.event_id,
            "topic": topic
        }

    def run_cloud_pipeline(self, date_str: str) -> Dict:
        """
        运行云端处理流水线
        """
        logger.info("=" * 50)
        logger.info("开始云端处理流水线")

        # 1. 批处理 ETL
        processed_count = self.batch_processor.run_etl_job(date_str)

        # 2. 数据挖掘
        filters = {"weather": "rain", "object_type": "construction_zone"}
        scenarios = self.data_miner.mine_scenarios(filters)
        logger.info(f"挖掘到 {len(scenarios)} 个场景")

        # 3. 数据标注（模拟）
        annotations = []
        for scenario in scenarios:
            can = CANSignal(
                longitudinal_acc=-5.0,
                lateral_acc=2.0,
                speed=80.0,
                speed_delta_1s=15.0,
                brake_pedal=30.0,
                steering_wheel_velocity=50.0
            )
            perception = PerceptionResult(
                object_confidence=0.35,
                ood_score=0.75,
                trajectory_error=1.5,
                entropy=2.0
            )
            event = VehicleEvent(
                event_id=f"EVT_{random.randint(1000, 9999)}",
                vehicle_id=f"VEH_{random.randint(10000, 99999)}",
                trigger_type=TriggerType.MODEL_BASED,
                trigger_reason="construction_zone_low_confidence",
                can_signal=can,
                perception_result=perception
            )
            annotation = self.annotation_pipeline.pre_annotation(event)
            annotations.append(annotation)

        # 4. 数据增强
        augmented_data = []
        for annotation in annotations:
            # 传统增强
            traditional = self.augmentation_engine.traditional_augmentation(annotation)
            augmented_data.extend(traditional)
            # AIGC 增强
            aigc = self.augmentation_engine.aigc_augmentation(annotation, "nerf")
            augmented_data.append(aigc)

        logger.info(f"生成 {len(augmented_data)} 条增强数据")

        return {
            "status": "completed",
            "etl_processed": processed_count,
            "scenarios_mined": len(scenarios),
            "annotations": len(annotations),
            "augmented_data": len(augmented_data)
        }

    def run_training_loop(self) -> Dict:
        """
        运行训练闭环
        """
        logger.info("=" * 50)
        logger.info("开始训练闭环")

        # 1. 构建数据集
        raw_data = [{"id": i} for i in range(100)]
        annotations = [{"id": i, "label": "car"} for i in range(80)]
        augmented = [{"id": i, "aug": True} for i in range(200)]

        dataset = self.training_pipeline.build_dataset(raw_data, annotations, augmented)

        # 2. 训练模型
        model = self.training_pipeline.train_model(dataset, "perception_model_v2")

        # 3. 离线评估
        metrics = self.training_pipeline.offline_evaluation(model, raw_data[:20])

        # 4. 闭环仿真
        scenarios = [{"id": i, "type": "construction"} for i in range(1000)]
        sim_result = self.training_pipeline.closed_loop_simulation(model, scenarios)

        # 5. 生成 OTA 包
        ota_package = self.training_pipeline.generate_ota_package(model, "2.1.0")

        return {
            "status": "completed",
            "dataset": dataset,
            "model": model,
            "metrics": metrics,
            "simulation": sim_result,
            "ota_package": ota_package
        }

    def run_full_pipeline(self) -> Dict:
        """
        运行完整闭环 Pipeline
        """
        logger.info("=" * 60)
        logger.info("启动自动驾驶数据闭环完整 Pipeline")
        logger.info("=" * 60)

        results = {}

        # 阶段1: 车端触发
        logger.info("\n=== 阶段1: 车端触发 ===")
        can = CANSignal(
            longitudinal_acc=-5.5,
            lateral_acc=2.5,
            speed=85.0,
            speed_delta_1s=18.0,
            brake_pedal=35.0,
            steering_wheel_velocity=80.0
        )
        perception = PerceptionResult(
            object_confidence=0.32,
            ood_score=0.78,
            trajectory_error=1.8,
            entropy=2.2
        )
        results["vehicle"] = self.process_vehicle_event(can, perception)

        # 阶段2: 云端处理
        logger.info("\n=== 阶段2: 云端处理 ===")
        results["cloud"] = self.run_cloud_pipeline("20260317")

        # 阶段3: 训练闭环
        logger.info("\n=== 阶段3: 训练闭环 ===")
        results["training"] = self.run_training_loop()

        logger.info("\n" + "=" * 60)
        logger.info("完整 Pipeline 执行完成")
        logger.info("=" * 60)

        return results

    def _create_metadata(self, event: VehicleEvent) -> Dict:
        """创建事件元数据"""
        return {
            "event_id": event.event_id,
            "vehicle_id": event.vehicle_id,
            "trigger_type": event.trigger_type.value,
            "trigger_reason": event.trigger_reason,
            "timestamp": event.timestamp.isoformat(),
            "can_data": {
                "longitudinal_acc": event.can_signal.longitudinal_acc,
                "lateral_acc": event.can_signal.lateral_acc,
                "speed": event.can_signal.speed,
                "speed_delta_1s": event.can_signal.speed_delta_1s,
                "brake_pedal": event.can_signal.brake_pedal,
                "steering_wheel_velocity": event.can_signal.steering_wheel_velocity
            },
            "perception": {
                "object_confidence": event.perception_result.object_confidence if event.perception_result else None,
                "ood_score": event.perception_result.ood_score if event.perception_result else None,
                "entropy": event.perception_result.entropy if event.perception_result else None
            } if event.perception_result else None
        }

    def _store_event_data(self, event: VehicleEvent, metadata: Dict, data_size: int):
        """存储事件数据"""
        self.storage.store_metadata(metadata)

        if event.perception_result:
            self.storage.store_timeseries_data(
                event.vehicle_id,
                metadata.get("can_data", {})
            )

        # 模拟存储非结构化数据
        location = {"lat": 22.926233, "lon": 114.200767}
        self.storage.store_graph_data(event.vehicle_id, location)


def main():
    """
    主函数 - 执行完整闭环
    """
    import sys

    orchestrator = AutopilotDataLoopOrchestrator()

    # 如果有参数，执行指定模式
    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "vehicle":
            # 车端触发测试
            can = CANSignal(
                longitudinal_acc=-5.5,
                lateral_acc=2.5,
                speed=85.0,
                speed_delta_1s=18.0,
                brake_pedal=35.0,
                steering_wheel_velocity=80.0
            )
            perception = PerceptionResult(
                object_confidence=0.32,
                ood_score=0.78,
                trajectory_error=1.8,
                entropy=2.2
            )
            result = orchestrator.process_vehicle_event(can, perception)

        elif mode == "cloud":
            # 云端处理测试
            result = orchestrator.run_cloud_pipeline("20260317")

        elif mode == "training":
            # 训练闭环测试
            result = orchestrator.run_training_loop()

        elif mode == "full":
            # 完整流程测试
            result = orchestrator.run_full_pipeline()

        else:
            # 无效参数，返回使用说明
            result = {
                "status": "error",
                "message": "无效模式。可用模式: vehicle, cloud, training, full"
            }
    else:
        # 默认执行完整流程
        result = orchestrator.run_full_pipeline()

    # 返回 JSON 格式结果
    output = {
        "result": "1.0",  # 成功
        "error": "0.0",   # 无错误
        "answer": json.dumps(result, ensure_ascii=False, indent=2)
    }

    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
