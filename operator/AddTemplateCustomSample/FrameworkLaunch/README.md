## 概述
本样例基于AddTemplateCustom算子工程，介绍了单算子工程、第三方框架调用。
## 目录结构介绍
```
├── FrameworkLaunch           //使用框架调用的方式调用Add算子
│   ├── AclNNInvocation       // 通过aclnn调用的方式调用AddTemplateCustom模板参数算子
│   ├── AclNNInvocationNaive  // 通过aclnn调用的方式调用AddTemplateCustom模板参数算子, 简化了编译脚本
│   ├── AclOfflineModel       // 通过aclopExecuteV2调用的方式调用AddTemplateCustom模板参数算子
│   ├── AclOnlineModel        // 通过aclopCompile调用的方式调用AddTemplateCustom模板参数算子
│   ├── AddTemplateCustom     // AddTemplateCustom模板参数算子工程
│   ├── CppExtensions         // Pybind方式调用AddTemplateCustom模板参数算子
│   ├── PytorchInvocation     // 通过pytorch调用的方式调用AddTemplateCustom模板参数算子
│   ├── TensorflowInvocation  // 通过tensorflow调用的方式调用AddTemplateCustom模板参数算子
│   └── AddCustom.json        // AddTemplateCustom模板参数算子的原型定义json文件
```
## 算子工程介绍
其中，模板参数算子工程目录AddTemplateCustom包含算子实现的模板文件、编译脚本等，如下所示:
```
├── AddTemplateCustom                          //Add自定义模板参数算子工程
│   ├── cmake
│   ├── framework                           // 算子插件实现文件目录，单算子模型文件的生成不依赖算子适配插件，无需关注
│   ├── op_host                             // host侧实现文件
│   │   ├── add_custom.cpp                  // host侧tiling定义
│   │   ├── add_custom_tiling.h             // host侧tiling定义头文件
│   │   ├── CMakeLists.txt                  // host侧的CMakeLists.txt
│   ├── op_kernel                           // kernel侧实现文件
│   │   ├── add_custom.cpp                  // kernel侧算子实现文件
│   │   ├── CMakeLists.txt                  // kernel侧的CMakeLists.txt
│   │   ├── tiling_key_add_custom.cpp       // kernel侧模板参数定义头文件
│   ├── scripts                             // 自定义算子工程打包相关脚本所在目录
│   ├── build.sh                            // 编译入口脚本
│   ├── CMakeLists.txt                      // 算子工程的CMakeLists.txt
│   └── CMakePresets.json                   // 编译配置项
```
CANN软件包中提供了工程创建工具msopgen，AddTemplateCustom算子工程可通过AddCustom.json自动创建，具体请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>算子开发>算子开发工程>基于自定义算子工程的算子开发>创建算子工程 章节。

算子实现的介绍请参考[AddTemplateCustom](./AddTemplateCustom/README.md)。
## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程生成算子安装包；
- 安装自定义算子到算子库中；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)完成源码包获取。
### 2. 编译算子工程<a name="operatorcompile"></a>
  编译自定义算子工程，构建生成自定义算子包。

  - 执行如下命令，切换到算子工程AddTemplateCustom目录。   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/AddTemplateCustomSample/FrameworkLaunch/AddTemplateCustom
    ```

  - 修改CMakePresets.json中ASCEND_CANN_PACKAGE_PATH为CANN软件包安装后的实际路径。


    ```json
    {
        ……
        "configurePresets": [
            {
                    ……
                    "ASCEND_CANN_PACKAGE_PATH": {
                        "type": "PATH",
                        "value": "/usr/local/Ascend/ascend-toolkit/latest"   //请替换为CANN软件包安装后的实际路径。eg:/home/HwHiAiUser/Ascend/ascend-toolkit/latest
                    },
                    ……
            }
        ]
    }
    ```
  - 在算子工程AddTemplateCustom目录下执行如下命令，进行算子工程编译。

    ```bash
    ./build.sh
    ```
编译成功后，会在当前目录下创建build_out目录，并在build_out目录下生成自定义算子安装包custom_opp_\<target os>_\<target architecture>.run，例如“custom_opp_ubuntu_x86_64.run”。

备注：如果要使用dump调试功能，需要移除op_host内和CMakePresets.json内的Atlas 训练系列产品、Atlas 200/500 A2 推理产品的配置。

### 3. 部署算子包

执行如下命令，在自定义算子安装包所在路径下，安装自定义算子包。
  ```bash
  cd build_out
  ./custom_opp_<target os>_<target architecture>.run
  ```
命令执行成功后，自定义算子包中的相关文件将部署至当前环境的OPP算子库的vendors/customize目录中。

### 4. 配置环境变量

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
### 5. 调用执行算子工程
- [aclnn调用AddTemplateCustom算子工程](./AclNNInvocation/README.md)
- [aclnn调用AddTemplateCustom算子工程(代码简化)](./AclNNInvocationNaive/README.md)
- [aclopExecuteV2模型调用AddTemplateCustom算子工程](./AclOfflineModel/README.md)
- [aclopCompile模型调用AddTemplateCustom算子工程](./AclOnlineModel/README.md)
- [cpp-extension模型调用AddTemplateCustom算子工程](./CppExtensions/README.md)
- [pytorch调用AddTemplateCustom算子工程](./PytorchInvocation/README.md)
- [tensorflow调用AddTemplateCustom算子工程](./TensorflowInvocation/AscendCustomToTensorFlowCustom/README.md)

## 更新说明
| 时间         | 更新事项       |
|------------|------------|
| 2024/10/25 | 新增模板参数算子样例 |