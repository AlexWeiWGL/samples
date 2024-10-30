## 概述
本样例介绍非对齐WholeReduceSum算子的核函数直调方法。
## 目录结构介绍
```
├── KernelLaunch                        // 使用核函数直调的方式调用非对齐WholeReduceSum自定义算子
│   └── WholeReduceSumKernelInvocation  // Kernel Launch方式调用核函数样例
```
## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)完成源码包获取。
### 2. 编译运行样例工程
- [WholeReduceSumKernelInvocation样例运行](./WholeReduceSumKernelInvocation/README.md)
## 更新说明
| 时间 | 更新事项 | 注意事项 |
| - | - | - |
| 2024/9/19 | 新增本readme | 无 |
