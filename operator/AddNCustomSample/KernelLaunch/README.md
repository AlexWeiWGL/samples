## 概述
本样例基于[AddCustom-AddCustomSample](../../AddCustomSample/KernelLaunch/)算子工程，介绍了单算子直调工程动态输入特性。
## 目录结构介绍
```
├── KernelLaunch                // 使用核函数直调的方式调用AddN自定义算子
│   ├── cmake                   // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── addn_custom.cpp          // 算子kernel实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
```
## 代码实现介绍
动态输入特性是指，核函数的入参采用ListTensorDesc的结构存储输入数据信息，对应的，调用框架侧需构造TensorList结构保存参数信息，具体如下：

框架侧：

- 构造TensorList数据结构，示例如下。

   ```cpp
   constexpr uint32_t SHAPE_DIM = 2;
    struct TensorDesc {
        uint32_t dim{SHAPE_DIM};
        uint32_t index;
        uint64_t shape[SHAPE_DIM] = {8, 2048};
    };

    constexpr uint32_t TENSOR_DESC_NUM = 2;
    struct ListTensorDesc {
        uint64_t ptrOffset;
        TensorDesc tensorDesc[TENSOR_DESC_NUM];
        uintptr_t dataPtr[TENSOR_DESC_NUM];
    } inputDesc;
   ```

- 将申请分配的Tensor入参组合成ListTensorDesc的数据结构，示例如下。

  ```cpp
  inputDesc = {(1 + (1 + SHAPE_DIM) * TENSOR_DESC_NUM) * sizeof(uint64_t),
                 {xDesc, yDesc},
                 {(uintptr_t)xDevice, (uintptr_t)yDevice}};
   ```

kernel侧:

- 按照框架侧传入的数据格式，解析出对应的各入参，示例如下。

  ```cpp
    uint64_t buf[SHAPE_DIM] = {0};
    AscendC::TensorDesc<int32_t> tensorDesc;
    tensorDesc.SetShapeAddr(buf);
    listTensorDesc.GetDesc(tensorDesc, 0);
    uint64_t totalLength = tensorDesc.GetShape(0) * tensorDesc.GetShape(1);
    __gm__ uint8_t *x = listTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    __gm__ uint8_t *y = listTensorDesc.GetDataPtr<__gm__ uint8_t>(1);
   ```

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/AddNCustomSample/KernelLaunch/
    ```
  - 配置环境变量

    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```

    配置仿真模式日志文件目录，默认为sim_log。
    ```bash
    export CAMODEL_LOG_PATH=./sim_log
    ```

  - 样例执行

    ```bash
    bash run.sh -r [RUN_MODE] -v  [SOC_VERSION]
    ```
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
      - Atlas 推理系列产品（Ascend 310P处理器）参数值：Ascend310P1、Ascend310P3
      - Atlas 训练系列产品参数值：AscendxxxA、AscendxxxB
      - Atlas A2训练系列产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

    示例如下。
    ```bash
    bash run.sh -r cpu -v Ascend310P1
    ```
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/10/01 | 新增直调方式动态输入样例 |