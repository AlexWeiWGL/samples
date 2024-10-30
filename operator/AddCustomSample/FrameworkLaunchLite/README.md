## 概述
本样例基于AddCustom算子工程，介绍了msOpGen工具生成简易自定义算子工程和单算子调用。
## 目录结构介绍
```
├── FrameworkLaunchLite       //使用框架调用的方式调用Add算子
│   ├── AclNNInvocationNaive  // 通过aclnn调用的方式调用AddCustom算子, 简化了编译脚本
│   ├── AddCustom             // AddCustom算子host和kernel实现
│   ├── install.sh            // 脚本，调用msOpGen生成简易自定义算子工程，并编译
│   └── AddCustom.json        // AddCustom算子的原型定义json文件
```
## 算子工程介绍
install.sh脚本会调用msopgen工具生成简易自定义算子工程，工具命令及生成工程目录结构介绍请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>附录>简易自定义算子工程 章节。

创建完简易自定义算子工程后，开发者重点需要完成算子工程目录CustomOp下host和kernel的功能开发。为简化样例运行流程，本样例已在AddCustom目录准备好了必要的算子实现，install.sh脚本会自动将实现复制到CustomOp对应目录下，再编译算子。

## 编译运行样例算子
针对简易自定义算子工程，编译运行包含如下步骤：
- 调用msOpGen工具生成简易自定义算子工程；
- 完成算子host和kernel实现；
- 编译算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)完成源码包获取。

### 2. 配置环境变量

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

### 3. 生成简易自定义算子工程，复制host和kernel实现并编译算子
  - 执行如下命令，切换到msOpGen脚本install.sh所在目录。   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/AddCustomSample/FrameworkLaunchLite
    ```

  - 调用脚本，生成简易自定义算子工程，复制host和kernel实现并编译算子。

    ```bash
    ./install.sh -v [SOC_VERSION]
    ```
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
        - Atlas 推理系列产品（Ascend 310P处理器）参数值：Ascend310P1、Ascend310P3
        - Atlas 训练系列产品参数值：AscendxxxA、AscendxxxB
        - Atlas A2训练系列产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

脚本运行成功后，会在当前目录下创建CustomOp目录，编译完成后，会在CustomOp/build_out/op_api/lib目录下生成自定义算子库文件libcust_opapi.so，在CustomOp/build_out/op_api/include目录下生成aclnn接口的头文件。

备注：如果要使用dump调试功能，需要移除op_host内和CMakeLists.txt内的Atlas 训练系列产品、Atlas 200/500 A2 推理产品的配置。

### 4. 调用执行算子工程
- [aclnn调用AddCustom算子工程(代码简化)](./AclNNInvocationNaive/README.md)
## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2024.10.21 | 初始版本                     |