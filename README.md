# datamining
车端数据挖掘pipeline






V2：
项目结构：autopilot_data_loop_v2/

├── __init__.py           # 包初始化

├── config.py            # 系统配置

├── vehicle_trigger.py   # 车端触发模块（三层漏斗）

├── cloud_edge.py        # 云端接入模块

├── stream_batch.py      # 流批一体处理

├── data_mining.py       # 数据挖掘与标注

├── training_validation.py # 模型训练与验证

├── main.py              # 主协调器

└── requirements.txt     # 依赖包

四大核心模块
1. 车端触发模块 (vehicle_trigger.py)
第一层: 硬规则触发 (急刹、急拐、速度差、接管等)
第二层: 轻量级感知异常检测 (低置信度、OOD检测、轨迹偏差)
第三层: 影子模式与不确定性估算 (决策分歧、熵值触发)
数据封装: 完整的元数据格式，包含位置、标签、文件清单等
2. 云端接入模块 (cloud_edge.py)
传输路由: 根据数据大小自动选择MQTT/gRPC/QUIC协议
网关认证: DID证书验证
Kafka Producer: 实现真正的producer->topic->consumer->consumer group流程
分级存储: S3/OSS (非结构化)、IoTDB (时序)、PostgreSQL (元数据)
3. 流批一体处理 (stream_batch.py)
Kafka Consumer Group: 真实的消费者组实现，支持负载均衡和故障转移
Flink Stream Processor: KeyBy分区、Window窗口、Process处理、Sink输出
Spark ETL Processor: 数据清洗、验证、转换流程
4. 数据挖掘与训练 (data_mining.py + training_validation.py)
场景挖掘: 规则过滤、K-Means聚类、难例挖掘
人机协同标注: 预标注、置信度分流、质量检查
数据增强: 传统增强 + AIGC (NeRF、Diffusion、仿真)
模型训练: DDP/FSDP分布式训练
OTA部署: 灰度发布机制
运行演示

cd autopilot_data_loop_v2
python3 main.py --demo
演示结果：

86个触发事件 (规则触发/模型触发/不确定性触发)
挖掘到40个场景
标注10个事件
生成9个增强数据
训练完成，mAP达到0.86
