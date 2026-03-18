# 设计模式

Read 车端检测任务_详细设计版.md
先出设计方案，确认后再开发。根据文档内容，设计一个完整的自动驾驶数据闭环系统，包括：

车端触发模块（三层触发）
数据流式回传云端模块（Kafka producer->topic->consumer group->consumer）
云端挖掘模块
模型训练模块

自动驾驶数据闭环系统 - 设计方案
一、整体架构

┌─────────────────────────────────────────────────────────────────────────────┐
│                           车端触发模块 (Vehicle Side)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │  Rule-based  │  │  Model-based │  │ Uncertainty  │                       │
│  │  Layer 1     │  │  Layer 2     │  │  Layer 3     │                       │
│  │  (硬规则)     │  │  (感知模型)   │  │  (影子/熵)   │                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
│         │                 │                 │                                 │
│         └─────────────────┴─────────────────┘                                 │
│                           │ 触发事件                                          │
│                    ┌──────▼──────┐                                         │
│                    │ 数据封装器  │                                         │
│                    │ DataPackage│                                         │
│                    └──────┬──────┘                                         │
└───────────────────────────┼─────────────────────────────────────────────────┘
                            │ 上传
┌───────────────────────────▼─────────────────────────────────────────────────┐
│                    数据流式回传云端模块 (Cloud Edge)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐               │
│  │   MQTT       │     │   gRPC       │     │   QUIC       │               │
│  │  (<100KB)    │     │ (<500MB)     │     │  (>500MB)    │               │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘               │
│         │                    │                    │                         │
│  ┌──────▼────────────────────┴────────────────────▼───────┐               │
│  │              网关认证 (DID 证书验证)                    │               │
│  └───────────────────────┬────────────────────────────────┘               │
│                          │                                                  │
│  ┌───────────────────────▼────────────────────────────────┐               │
│  │              Kafka 消息队列                           │               │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐            │               │
│  │  │   Topic  │  │   Topic  │  │   Topic  │            │               │
│  │  │Metadata  │→  │High Prio │→  │Med Prio │            │               │
│  │  └──────────┘  └──────────┘  └──────────┘            │               │
│  └───────────────────────┬────────────────────────────────┘               │
│                          │                                                  │
└──────────────────────────┼─────────────────────────────────────────────────┘
                            │ 消费
┌──────────────────────────▼─────────────────────────────────────────────────┐
│                   流批一体数据处理 (Processing)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐           ┌───────────────────┐                   │
│  │  Flink 流处理     │           │  Spark 批处理     │                   │
│  │  - 实时元数据解析  │           │  - 数据清洗       │                   │
│  │  - 车辆健康大屏   │           │  - 去重对齐       │                   │
│  │  - 实时告警       │           │  - ETL 作业       │                   │
│  └─────────┬─────────┘           └─────────┬─────────┘                   │
│            │                               │                             │
│  ┌─────────▼───────────────────────────────▼─────────┐                   │
│  │              分级存储                         │                   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐    │                   │
│  │  │   OSS    │  │  IoTDB   │  │PostgreSQL│    │                   │
│  │  │非结构化  │  │  时序    │  │  元数据   │    │                   │
│  │  └──────────┘  └──────────┘  └──────────┘    │                   │
│  └──────────────────────────────────────────────┘                   │
└──────────────────────────┬─────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼─────────────────────────────────────────────────┐
│                   数据挖掘与构建模块 (Mining & Construction)               │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │  规则过滤    │→ │  聚类分析    │→ │  难例挖掘    │                      │
│  └──────────────┘  └──────────────┘  └──────────────┘                      │
│                            ↓                                              │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │              人机协同标注                                 │            │
│  │  预标注 → 人工修正 → 置信度分流 → 专家重演               │            │
│  └───────────────────────┬───────────────────────────────────┘            │
│                            ↓                                              │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │              高级数据增强                                 │            │
│  │  传统增强 + NeRF/3DGS + Diffusion + 仿真泛化              │            │
│  └───────────────────────┬───────────────────────────────────┘            │
└──────────────────────────┼─────────────────────────────────────────────────┘
                            │
┌──────────────────────────▼─────────────────────────────────────────────────┐
│                 模型训练与验证模块 (Training & Validation)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      │
│  │  模型训练    │→ │  离线评估    │→ │  闭环仿真    │                      │
│  │  (DDP/FSDP)  │  │  (mAP,Recall)│  │  (Carla/VTD) │                      │
│  └──────────────┘  └──────────────┘  └──────┬───────┘                      │
│                                             ↓                               │
│  ┌───────────────────────────────────────────────────────────┐            │
│  │              OTA 部署                                     │            │
│  │  影子模式 → 灰度推送 → 全量发布                          │            │
│  └───────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
二、模块详细设计
模块1：车端触发模块 (VehicleTriggerModule)
类名	职责
RuleBasedTrigger	硬规则触发（CAN 信号监测）
ModelBasedTrigger	模型触发（置信度/OOD/轨迹）
UncertaintyTrigger	不确定性触发（影子模式/熵）
DataPackageBuilder	数据封装（Metadata + 原始数据）
触发流程：


CAN信号 → Layer1规则 → Layer2模型 → Layer3不确定性 → 触发/不触发
                                                    ↓
                                            DataPackageBuilder
                                                    ↓
                                            (元数据+原始数据)
模块2：数据流式回传云端模块 (CloudEdgeModule)
类名	职责
TransmissionRouter	数据大小路由（MQTT/gRPC/QUIC）
CloudGateway	DID 证书认证
KafkaProducer	Kafka 生产者（发送元数据）
StorageManager	OSS/IoTDB/PG 存储
Kafka 消息流：


车端 → KafkaProducer → topic_metadata → Consumer Group
                                         ↓
                                    Consumer 1
                                    Consumer 2
                                    Consumer 3
                                         ↓
                                    处理/存储
模块3：流批一体数据处理 (StreamBatchProcessor)
类名	职责
FlinkStreamProcessor	实时流处理（元数据解析/大屏/告警）
SparkETLProcessor	离线批处理（清洗/去重/对齐）
TieredStorage	分级存储（OSS/IoTDB/PG）
Flink Pipeline：


Kafka Source → Parse → Map → KeyBy → Window → Process → Sink
                                                      ↓
                                              车辆健康大屏
模块4：数据挖掘与构建模块 (DataMiningModule)
类名	职责
ScenarioMiner	场景挖掘（规则/聚类/难例）
AnnotationPipeline	人机协同标注（预标注/人工/分流）
DataAugmenter	数据增强（传统/NeRF/Diffusion/仿真）
标注流程：


原始数据 → 预标注 → 置信度>0.9 → 免检入库
                    ↓
               0.4<置信度≤0.9 → 人工抽检
                    ↓
               置信度≤0.4 → 人工全检
模块5：模型训练与验证模块 (TrainingValidationModule)
类名	职责
ModelTrainer	模型训练（DDP/FSDP）
ModelEvaluator	离线评估（mAP/Recall/ADE）
SimulatorRunner	闭环仿真（Carla/VTD）
OTADeployer	OTA 部署（影子/灰度/全量）
三、核心数据结构
DataPackage（触发数据包）

@dataclass
class DataPackage:
    event_id: str
    vehicle_id: str
    trigger_type: str
    trigger_time: datetime
    location: Dict
    scenario_tags: List[str]
    trigger_details: Dict
    sensor_status: Dict
    file_manifest: List[Dict]  # 文件清单
    raw_data: Dict  # 原始数据（视频/点云/CAN）
KafkaMessage

{
    "event_id": "evt_xxx",
    "vehicle_id": "VIN_xxx",
    "trigger_type": "xxx",
    "timestamp": "2026-03-18T03:30:55Z",
    "metadata": {...},
    "oss_path": "s3://..."
}
四、技术栈
功能	技术
消息队列	confluent-kafka (Kafka Python)
流处理	pyflink (Apache Flink)
批处理	pyspark (Apache Spark)
对象存储	boto3 (AWS S3)
时序库	influxdb-client (InfluxDB)
关系数据库	psycopg2 (PostgreSQL)
深度学习	pytorch (PyTorch)



五、模块化设计
车端触发模块 (VehicleTriggerModule)

RuleBasedTrigger
ModelBasedTrigger
UncertaintyTrigger
DataPackageBuilder
数据流式回传云端模块 (CloudEdgeModule)

TransmissionRouter
CloudGateway
KafkaProducer
StorageManager
流批一体数据处理 (StreamBatchProcessor)

FlinkStreamProcessor
SparkETLProcessor
TieredStorage
数据挖掘与构建模块 (DataMiningModule)

ScenarioMiner
AnnotationPipeline
DataAugmenter
模型训练与验证模块 (TrainingValidationModule)

ModelTrainer
ModelEvaluator
SimulatorRunner
OTADeployer

