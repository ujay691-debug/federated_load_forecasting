项目目录如下：

federated_load_forecasting/
│
├── data/
│   ├── client_1.csv
│   ├── client_2.csv
│   ├── ...
│   └── client_9.csv
│
├── models/
│   └── cnn_lstm.py
│
├── utils/
│   ├── data_utils.py
│   ├── metrics.py
│   └── aggregation.py
│
├── client.py
├── server.py
├── federated_main.py
├── centralized_main.py
└── config.py

一、先做的事
1. 把你的 9 个客户端 csv 放到 data 目录
2. 文件名改成 client_1.csv 到 client_9.csv
3. 确保 9 份数据至少有：
   - timestamp
   - 目标列，比如 total_load 或 gc
4. 如果启用了气象特征，还要有对应列：
   - temp2m_c 或 temp2m_k
   - wind10m_ms
   - ghi_wm2
   - rh2m_pct（如果 use_rh=True）

二、最重要的配置位置
打开 config.py，主要改这里：
1. data.target_col
2. data.seq_len
3. data.horizon
4. feature 里的各特征开关
5. train.epochs
6. federated.rounds
7. federated.local_epochs

三、运行方式
在 PyCharm 终端或 Windows 命令行里进入项目根目录后运行：

联邦学习：
python federated_main.py

中心化对比：
python centralized_main.py

四、结果输出
联邦结果默认保存在：
runs/federated/

中心化结果默认保存在：
runs/centralized/

五、当前实现逻辑
1. 联邦训练采用最普通的 FedAvg
2. 每个客户端本地训练同一个 CNN-LSTM
3. 服务器只聚合参数，不读取本地原始负荷数据
4. 测试时，各客户端分别预测，再按 timestamp 对齐求和，得到区域总负荷预测

六、说明
当前这版为了保证结构清晰，区域聚合评估默认以各客户端预测时间戳对齐后求和。
只要 9 个客户端的时间戳体系一致，这个流程就能直接跑通。
