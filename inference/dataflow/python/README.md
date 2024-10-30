# Python样例使用指导
## 目录结构
.   
├── README.md      
├── config   
│   ├── add 用例中tensorflow计算图文件,可以通过添加后缀pb打开查看   
│   ├── add_func.json 用例中udf配置文件    
│   ├── add_graph.json 用例中计算图编译配置文件  
│   ├── data_flow_deploy_info.json 用例中部署位置配置文件  
│   └── invoke_func.json 用例中udf_call_nn的编译配置文件  
├── sample1.py 样例1展示了基本的DataFlow API构图，包含UDF，GraphPp和UDF执行NN推理三种类型节点的构造和执行    
├── sample2.py 样例2展示了python dataflow调用 udf python的过程   
├── sample3.py 样例3展示了使能异常上报的样例   
├── test_perf.py 性能打点样例  
└── udf_py   
    ├── udf_add.py 使用python实现udf多func功能  
    └── udf_control.py 使用python是udf功能，用于控制udf_add中多func实际执行的func  


## 环境准备
参考[环境准备](../../../README.md#环境准备)下载安装驱动/固件/CANN软件包   
python 版本要求：python3.9


## 运行样例
```bash
# 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       #0 debug 1 info 2 warn 3 error 不设置默认error级别
export ASCEND_SLOG_PRINT_TO_STDOUT=1   # 日志打屏，不设置日志落盘默认路径
# 必选
# 此处以ASCEND_INSTALL_PATH配置以CANN开发套件实际安装路径为准
export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit
source ${ASCEND_INSTALL_PATH}/latest/bin/setenv.bash
export RESOURCE_CONFIG_PATH=xxx/xxx/xxx/numa_config.json # 环境中实际的numa_config文件位置

python3.9 sample1.py
python3.9 sample2.py
python3.9 sample3.py
python3.9 test_perf.py
```

