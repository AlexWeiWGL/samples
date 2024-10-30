## 目录

- [样例介绍](#样例介绍)
- [环境准备](#环境准备)
- [样例运行](#样例运行)


## 样例介绍

功能：通过LLM-DataDist接口实现分离部署场景下KvCache的管理功能。

| 目录名称                                                   | 功能描述                                             |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [basic_api_samples](./basic_api_samples) | LLM-DataDist基础接口样例 |
| [basic_api_samples](./cache_manager_api_samples) | LLM-DataDist cache manager接口样例 |


## 环境准备
Prompt代码中使用了torchair将kv_cache的tensor地址转为torch tensor并赋值，所以需要安装torch_npu相关包
当前只支持python3.9版本

## 样例运行
- 运行basic_api_samples样例
    - 执行前准备：

      - 在Prompt与Decoder的主机分别以下命令，查询该主机的device ip信息
        ```
        for i in {0..7}; do hccn_tool -i $i -ip -g; done
        ```
        **注: 如果出现hccn_tool命令找不到的情况，可在CANN包安装目录下搜索hccn_tool，找到可执行文件执行**
      - 更改脚本中的device信息
        - prompt.py中，将DEVICE_IP_LIST中的device_ip修改为Prompt主机的各device_ip
        - decoder.py中，将PROMPT_CLUSTER_ID_TO_DEVICE_IP_LIST中的value修改为Prompt主机的各device_ip，将DECODER_DEVICE_IP_LIST修改为Decoder主机的各device_ip
        - 如果需要在同一个主机执行，可以修改脚本中的DEVICE_ID_LIST变量，将Prompt与Decoder分配到不同的device上(如: prompt配置为[0, 1, 2, 3], decoder配置为[4, 5, 6, 7])，同步修改关联的device ip
    - 执行样例程序：    
    分别在Prompt主机与Decoder主机，执行prompt.py与decoder.py，执行样例程序：
      ```
      # Prompt主机:
      python prompt.py
      # Decoder主机:
      python decoder.py
      ```
      如果需要使用多卡，则需分别拉起多个进程执行，并提供rank_id参数，以分别使用2个device为例：
      ```
      # Prompt主机:
      python prompt.py --rank_id=0 > prompt.log.0 2>&1 &
      python prompt.py --rank_id=1 > prompt.log.1 2>&1 &
      # Decoder主机:
      python decoder.py --rank_id=0 > decoder.log.0 2>&1 &
      python decoder.py --rank_id=1 > decoder.log.1 2>&1 &
      ```
- 运行cache_manager_api_samples样例
    - 执行前准备：

      - 在Prompt与Decoder的主机分别以下命令，查询该主机的device ip信息
        ```
        for i in {0..7}; do hccn_tool -i $i -ip -g; done
        ```
        **注: 如果出现hccn_tool命令找不到的情况，可在CANN包安装目录下搜索hccn_tool，找到可执行文件执行**
      - 更改脚本中的device信息
        - 将DEVICE_IP_LIST中的device_ip修改为本地主机的各device_ip
        - 将REMOTE_IP_LIST中的device_ip修改为远程主机的各device_ip
    - 执行pull cache样例程序，此样例程序展示了配置内存池场景下，使用allocate_cache并从远端pull：
    分别在Prompt主机与Decoder主机，执行样例程序：
      ```
      # Prompt主机:
      python pull_cache_sample.py --device_id 0 --cluster_id 1
      # Decoder主机:
      python pull_cache_sample.py --device_id 0 --cluster_id 2
      ```
    - 执行pull blocks样例程序，此样例程序使用torch自行申请内存，并从远端pull：
      分别在Prompt主机与Decoder主机，执行样例程序：
      ```
      # Prompt主机:
      python pull_blocks_sample.py --device_id 0 --cluster_id 1
      # Decoder主机:
      python pull_blocks_sample.py --device_id 0 --cluster_id 2
      ```

