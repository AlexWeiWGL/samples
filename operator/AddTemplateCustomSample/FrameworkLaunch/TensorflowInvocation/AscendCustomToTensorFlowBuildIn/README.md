## 概述
本样例展示了如何使用Ascend C自定义算子AddTemplateCustom映射到TensorFlow自定义算子Add，并通过TensorFlow调用Ascend C算子。

## 运行样例算子
### 1.编译算子工程
运行此样例前，请参考[编译算子工程](../../README.md#operatorcompile)完成前期准备。
### 2.tensorflow调用的方式调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/AddTemplateCustomSample/FrameworkLaunch/TensorflowInvocation/AscendCustomToTensorFlowBuildIn
    ```

  - 样例执行(tensorflow1.15)

    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比cpu和aicore运行结果。具体过程可参见run_add_custom.py脚本。
    ```bash
    python3 run_add_custom.py
    ```
  - 样例执行(tensorflow2.x)

    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比cpu和aicore运行结果。具体过程可参见run_add_custom_tf2.py脚本。
    ```bash
    python3 run_add_custom_tf2.py
    ```

    用户亦可参考run.sh脚本进行编译与运行。
    ```bash
    bash run.sh
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/10/25 | 新增模板参数算子样例 |