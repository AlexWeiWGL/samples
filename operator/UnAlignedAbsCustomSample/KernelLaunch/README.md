## 概述
本样例介绍无DataCopyPad的非对齐Abs算子核函数直调方法。
## 目录结构介绍
```
├── KernelLaunch                      // 使用核函数直调的方式调用非对齐Abs自定义算子
│   ├── AbsDuplicateKernelInvocation  // Kernel Launch方式调用非对齐Abs核函数样例，使用Duplicate配合mask清零
│   |── AbsGatherMaskKernelInvocation // Kernel Launch方式调用非对齐Abs核函数样例，使用GatherMask搬运
│   |── AbsPadKernelInvocation        // Kernel Launch方式调用非对齐Abs核函数样例，使用Pad清零
|   └── AbsUnPadKernelInvocation      // Kernel Launch方式调用非对齐Abs核函数样例，使用UnPad去除冗余值
```

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)完成源码包获取。
### 2. 编译运行样例工程
- [AbsGatherMaskKernelInvocation样例运行](./AbsGatherMaskKernelInvocation/README.md)
- [AbsDuplicateKernelInvocation样例运行](./AbsDuplicateKernelInvocation/README.md)
- [AbsPadKernelInvocation样例运行](./AbsPadKernelInvocation/README.md)
- [AbsUnPadKernelInvocation样例运行](./AbsUnPadKernelInvocation/README.md)
## 更新说明
| 时间| 更新事项|
| - | - |
| 2024/09/09 | 新增本Readme
