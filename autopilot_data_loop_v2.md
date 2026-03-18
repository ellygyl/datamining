# 任务目标
你是一个自动驾驶数据闭环（Data Closed-Loop）系统的架构师 Agent。你的任务是根据输入的多源异构数据（车辆 CAN 总线、传感器原始数据、高精地图、V2X 信息），设计并生成一套完整的**“车端异常触发 -> 云端挖掘 -> 模型训练”**的自动化 Pipeline 代码。

该 Pipeline 需覆盖从车端触发策略、云端高并发接入、数据清洗构建、人机协同标注、高级数据增强到模型训练验证的全流程。

# 步骤说明
1. 需要先出整体的设计方案
- 整体分为4个模块：
  - 车端触发模块，分为三层触发，分别封装
    - 第一层：规则触发 (Rule-based)
    - 第二层：模型触发 (Model-based)
    - 第三层：不确定性触发 (Uncertainty-based)
  - 数据流式回传云端模块：包含流式数据处理，并将处理后的数据离线存储
  - 云端挖掘模块：包含数据挖掘、标注、数据构建增强
  - 模型训练模块：包含模型训练、验证、迭代训练
2. 确认方案无误后，再生成云端和车端的代码

核心的业务逻辑内容如下：
## 1. 车端触发模块：
车端运行轻量级监控栈，通过“三级漏斗”机制筛选高价值异常数据（Corner Cases），仅上传触发片段（Snippet）及统计指标。

### 1.1 触发逻辑
- **第一层：硬规则触发 (Rule-based)**
  监测 CAN 总线信号，满足任一条件即触发：
  - 纵向加速度 < -4.0 m/s² (急刹) 且 车速 > 10 km/h
  - 横向加速度 > 3.0 m/s² (急拐) 且 车速 > 10 km/h
  - 速度差 Δv > 10 km/h (1s 内) 且 制动踏板开度 > 20%
  - 速度差 Δv > 30 km/h (1s 内) 且 制动踏板开度 > 40%
  - 方向盘角速度突变 或 制动踏板开度 > 80% (AEB 介入前兆)
  
- **第二层：轻量级感知异常检测 (Model-based)**
  运行量化模型 (INT8/FP16, TensorRT/OpenVINO 加速)：
  - **目标检测置信度低**：YOLOv5s/Transformer 变体检测到障碍物但 max_confidence < 0.4。
  - **分布外检测 (OOD)**：基于 AutoEncoder 重构误差或 One-class SVM，识别从未见过的障碍物（如异形车辆、掉落货物）。
  - **轨迹预测偏差**：简化 LSTM/GRU 预测轨迹与实际传感器观测轨迹误差 > 阈值。

- **第三层：影子模式与不确定性估算 (Shadow & Uncertainty)**
  - **影子模式 (Shadow Mode)**：并行运行“量产模型”与“最新影子模型”。若两者决策分歧（如：量产保持车道 vs 影子建议变道），强制触发记录。
  - **不确定性量化**：使用 Monte Carlo Dropout 或贝叶斯神经网络计算输出熵 (Entropy)。若熵值 > 阈值（模型“不知道”这是什么），即使无规则触发也上传。

- **触发数据封装示例**：
  - 元数据 (Metadata)，一个示例如下
    {
    "event_id": "evt_20260318_113055_a7f3b2",
    "vehicle_id": "VIN_AD_TEST_007",
    "trigger_type": "LOW_CONFIDENCE_SHADOW_MISMATCH", 
    "trigger_time_utc": "2026-03-18T03:30:55.123Z",
    "location": {
      "latitude": 39.9042,
      "longitude": 116.4074,
      "altitude": 52.5,
      "heading": 85.5,
      "road_name": "G4_Jinggangao_Expressway",
      "lane_id": "LN_G4_North_03"
    },
    "scenario_tags": [
      "highway",
      "construction_zone",
      "static_obstacle",
      "low_visibility",
      "human_takeover"
    ],
    "trigger_details": {
      "rule_hit": "shadow_mode_divergence",
      "production_model_decision": "KEEP_LANE (speed: 80km/h)",
      "shadow_model_decision": "CHANGE_LANE_LEFT (speed: 60km/h)",
      "perception_max_confidence": 0.32, 
      "uncertainty_entropy": 0.85,
      "ttc_to_obstacle": 2.4 
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
        "format": "h264",
        "size_bytes": 45020100,
        "s3_path": "s3://ad-data-lake/raw/2026/03/18/VIN_AD_TEST_007/evt_.../cam_front_long_00.bag"
      },
      {
        "file_name": "lidar_front_pointcloud.bin",
        "type": "pointcloud_raw",
        "format": "pcap/bin",
        "size_bytes": 12050000,
        "s3_path": "s3://ad-data-lake/raw/2026/03/18/VIN_AD_TEST_007/evt_.../lidar_front_pointcloud.bin"
      },
      {
        "file_name": "can_bus_signals.csv",
        "type": "vehicle_state",
        "format": "csv",
        "size_bytes": 102400,
        "s3_path": "s3://ad-data-lake/raw/2026/03/18/VIN_AD_TEST_007/evt_.../can_bus_signals.csv"
      },
      {
        "file_name": "planning_log.json",
        "type": "system_log",
        "format": "json",
        "size_bytes": 51200,
        "s3_path": "s3://ad-data-lake/raw/2026/03/18/VIN_AD_TEST_007/evt_.../planning_log.json"
      }
    ]
  }
  - 传感器原始数据 
    - 视频流 cam_front_long_00.bag，30秒的 H.264/H.265 编码视频。作用：用于视觉感知模型的重新训练和标注（画框、分割）。
    - 激光雷达点云 lidar_front_pointcloud.bin，每秒10-20帧的 XYZI (坐标+强度) 或 XYZIR (坐标+强度+回波次数) 数据。作用：用于3D检测、占用网络（Occupancy Network）训练，精确还原障碍物的几何形状（如施工围挡的不规则形状）
    - 毫米波雷达 radar_front.raw，点迹列表（距离、速度、角度、RCS），作用：用于测速校验和恶劣天气下的融合感知

  - 车辆状态与日志 
    - 车辆状态：can_bus_signals.csv，示例如下
    | timestamp_ms | vehicle_speed_kmh | longitudinal_accel_mps2 | lateral_accel_mps2 | brake_pedal_pos_pct | steering_angle_deg | autopilot_status | takeover_request |
    | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
    | 1710732645000 | 82.5 | -0.2 | 0.1 | 0.0 | 1.2 | ENGAGED | FALSE |
    | 1710732645100 | 82.4 | -0.3 | 0.1 | 0.0 | 1.3 | ENGAGED | FALSE |
    | ... | ... | ... | ... | ... | ... | ... | ... |
    | 1710732655123 | 78.0 | -2.5 | 0.5 | 15.0 | 5.5 | ENGAGED | TRUE |
    | 1710732655223 | 75.0 | -4.2 | 0.8 | 45.0 | 12.0 | DISENGAGED | TRUE |
    | ... | ... | ... | ... | ... | ... | ... | ... |
    - 规划日志：planning_log.json，一个示例如下
      {
      "timestamp": "2026-03-18T03:30:55.123Z",
      "perception_output": {
        "objects": [
          {
            "id": 101,
            "class": "UNKNOWN_STATIC",
            "bbox_3d": [15.2, 0.5, 45.0],
            "confidence": 0.32, 
            "velocity": [0, 0, 0]
          }
        ]
      },
      "prediction_output": {
        "trajectory_101": "STATIC"
      },
      "planning_decision": {
        "current_lane": "LANE_03",
        "target_speed": 80.0,
        "obstacle_avoidance_plan": "NONE", 
        "reason": "Object confidence (0.32) below threshold (0.40), treating as noise."
      },
      "shadow_model_decision": {
        "target_speed": 60.0,
        "plan": "LANE_CHANGE_LEFT",
        "reason": "New model identifies object as 'CONSTRUCTION_BARRIER' with conf 0.89."
      },
      "control_command": {
        "throttle": 0.0,
        "brake": 0.15,
        "steering": 5.5
      }
    }

## 2. 数据流式回传云端模块

### 2.1 先根据数据大小判断传输协议选择
- **小包/信令 (< 100 KB)**：MQTT over TLS (QoS 1)，用于实时状态上报。
- **中包/片段 (< 500 MB)**：gRPC/HTTP2，支持断点续传，用于常规异常片段。
- **大包/原始数据 (> > 500 MB)**：基于 QUIC (UDP) 的自定义私有协议，优化弱网环境下的视频/点云上传。

### 2.2 中心云架构
- **接入层**：输入
- **网关认证**：Apache Pulsar Gateway 或 Nginx + Lua，校验车辆 DID 证书，拦截非法请求。
- **消息缓冲**：Apache Kafka 作为“蓄水池”，解耦接入与处理，应对流量洪峰。注意，这部分要真的写出kafka的producer->topic->consumer group->consumer过程
- **分级存储**：
  - **非结构化数据** (视频/点云/雷达)：写入对象存储 (Aliyun OSS / AWS S3)，路径规范：`bucket/date/vehicle_id/event_id/`。
  - **时序数据** (CAN/GPS)：写入 Apache IoTDB 或 InfluxDB。
  - **元数据/索引** (事件类型/标签/状态)：写入 PostgreSQL 或 Elasticsearch。
  - **图数据** (路网拓扑)：写入 Neo4j 或 PostgreSQL (PostGIS)。

## 2. 云端：流批一体数据处理
- **实时流处理 (Real-time)**：Apache Flink 消费 Kafka 消息。
  - 实时解析元数据，更新车辆健康大屏，解析 JSON，提取关键标签
  - 实时告警：若发现高危场景（如 AEB 频繁触发），立即通知安全团队。
  - 自动提取关键帧，触发异步标注任务。
- **离线批处理 (Batch/ETL)**：Spark / PySpark。
  - 夜间执行大规模数据清洗、去重、时间同步对齐。

## 3. 数据挖掘与构建模块 (Data Mining & Construction)
### 3.1 高价值场景库构建
- **智能挖掘**：
  - **规则过滤**：SQL 查询特定场景（如 `weather='rain' AND object_type='construction_zone'`）。
  - **聚类分析**：对场景特征向量进行 K-Means/DBSCAN 聚类，挖掘长尾场景（Long-tail Cases）。
  - **难例挖掘 (Hard Mining)**：自动筛选上一代模型 Loss 最高或预测错误的样本。
- **数据清洗**：
  - 剔除传感器故障、时间戳不同步、画面模糊数据。
  - 多传感器时空同步校准（LiDAR-Camera Extrinsics Calibration）。

### 3.2 人机协同标注 (Human-in-the-Loop Annotation)
- **预标注 (Pre-annotation)**：利用云端大模型（Foundation Model）对回传数据进行自动标注。
- **人工修正**：标注员仅需修正错误部分（Correction Mode），效率提升 5-10 倍。
- **标注内容体系**：
  - **感知层**：2D/3D 包围盒（类别、属性、遮挡、截断）、语义分割（车道、路面、可行驶区）、多目标跟踪 (MOT ID)。
  - **预测层**：未来轨迹 (Future Trajectory)、驾驶意图 (Intent)、交互关系。
  - **规控层 (关键)**：针对接管场景，进行**专家重演 (Expert Demonstration)**，标注“理想轨迹”和“最优决策”，用于模仿学习。
- **置信度分流**：
  - 预标注置信度 >0.9：直接入库（免检）。
  - 0.4 < 置信度 <= 0.9：人工抽检。
  - 置信度 <= 0.4：人工全量复核。

### 3.3 高级数据增强 (Advanced Augmentation)
- **传统增强**：几何变换、光照调整、噪声注入、天气模拟（雨/雪/雾叠加）。
- **生成式增强 (AIGC)**：
  - **NeRF / 3D Gaussian Splatting**：重建真实场景，改变视角、时间（昼/夜）、天气，重新渲染并自动生成完美 GT。
  - **Diffusion Models**：In-painting 生成稀有障碍物（如侧翻车、动物群），Copy-Paste 到正常场景中。
- **仿真泛化**：将真实场景参数化 (OpenSCENARIO)，在仿真器中微调参数生成万级变体。

### 3.4 数据集构建与质量评估
- **数据集划分**：按比例构建 Train/Val/Test 集，确保 Corner Case 在测试集中有足够权重。
- **质量监控**：计算标注一致性指标 (IoU, Kappa)，定期审计。

## 4. 模型训练与验证闭环 (Training & Validation Loop)
*(此部分为新增核心环节，确保数据能转化为模型能力)*
- **模型训练**：
  - 加载预训练权重，使用构建好的数据集进行 Fine-tuning 或 End-to-End 训练。
  - 支持分布式训练 (PyTorch DDP/FSDP)，监控 Loss 曲线。
- **离线评估**：
  - 在隔离测试集上计算 mAP, Recall, ADE/FDE 等指标。
  - 对比基线模型，确认性能提升。
- **闭环仿真 (Closed-loop Simulation)**：
  - 将新模型部署到仿真平台（如 Carla, VTD, 或自研仿真器）。
  - **回归测试**：重跑所有历史失败案例，确保 Bug 已修复。
  - **压力测试**：在百万级随机场景中测试通过率。
- **OTA 部署**：
  - 仿真通过后，生成 OTA 包，灰度推送到车队影子模式运行，最终全量发布。

# 示例场景流程
**场景**：某重卡在高速公路上遭遇前方不规则施工围挡，感知系统置信度低，驾驶员接管。

1.  **车端触发**：
    - 影子模式检测到“量产模型”未识别围挡，而“影子模型”识别为障碍物，产生决策分歧。
    - 同时，感知模型输出熵值过高（Uncertainty High）。
    - 域控制器截取前后 60s 数据（摄像头视频、LiDAR 点云、CAN 总线），打包压缩。
2.  **数据上传**：
    - 元数据 (JSON) 经 MQTT 发送至 Kafka `topic_metadata`。
    - 大数据包 (Bin) 经 HTTP2 分片上传至 OSS `bucket_raw_data`。
3.  **云端处理**：
    - **Flink** 消费元数据，标记该车辆为“重点关注”，并在大屏告警。
    - **Trigger 函数** 监听 OSS 新文件，启动 **Spark ETL** 任务进行清洗和时空对齐。
    - **标注流水线** 调用大模型进行预标注，发现围挡置信度低，自动分发至人工标注台进行“专家重演”标注（绘制理想绕行轨迹）。
    - **增强模块** 利用 NeRF 将该场景转换为“夜间+雨天”版本，生成 50 条合成数据。
4.  **训练与验证**：
    - 将原始数据 + 标注数据 + 增强数据合并为 `dataset_construction_v2`。
    - 启动训练任务，Fine-tune 感知与规划模型。
    - **仿真系统** 自动回放该施工场景及 1000 个变体，新模型通过率从 40% 提升至 98%。
5.  **应用**：
    - 模型通过验收，生成 OTA 补丁。
    - 次日凌晨，向同车型车队推送更新，解决施工场景识别问题。

# 限制条件
1.  **名称**：请自定义一个专业的技能名称（例如 `AutopilotDataLoopOrchestrator`）。
2.  **代码语言**：所有逻辑脚本必须使用 **Python** 编写。
3.  **代码结构**：代码应体现模块化设计，包含数据接入、处理、标注调度、训练触发等类或函数。可以使用伪代码表示具体的深度学习框架调用（如 `model.train()`），但流程控制逻辑需完整。
  - 模块化设计 (Classes/Functions)。
  - 车端代码模拟触发逻辑和数据封装。
  - 云端代码模拟 Kafka 消费者、Flink 逻辑 (用 Python 伪代码表示)、Spark ETL 骨架、训练调度器。
  - 使用 boto3 模拟 S3 操作，pymongo 或 psycopg2 模拟数据库，confluent-kafka 模拟消息队列。
4.  **返回格式**：严格遵守下方的 JSON 格式，不要包含 Markdown 代码块标记以外的任何多余文本。

# 技能返回格式
每个环节交互请按照如下格式返回：
{
    "result": "0.0",
    "error": "0.0",
    "answer": "这里填写生成的完整 Python 代码字符串，注意转义换行符和引号"
}
