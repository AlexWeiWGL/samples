## AddN自定义算子样例说明
本样例通过Ascend C编程语言实现了AddN算子，并按照不同的算子调用方式分别给出了对应的端到端实现。
- [FrameworkLaunch](./FrameworkLaunch)：使用框架调用AddN自定义算子，算子输入为动态输入方式。  
  基于[AddCustomSample-FrameworkLaunch](../AddCustomSample/FrameworkLaunch/)样例工程，完成工程创建，在此工程基础上，实现自定义算子动态输入特性。
- [KernelLaunch](./KernelLaunch)：使用核函数直调AddN自定义算子，算子输入为动态输入方式。  
  基于[AddCustomSample-KernelLaunch](../AddCustomSample/KernelLaunch/)样例工程，完成工程创建，在此工程基础上，实现直调算子动态输入特性。

本样例中包含如下调用方式：
<table>
    <th>调用方式</th><th>描述</th>
    <tr>
        <!-- 列的方向占据2个cell -->
        <td rowspan='1'><a href="./FrameworkLaunch"> FrameworkLaunch</td><td>通过aclnn调用的方式调用动态输入AddNCustom算子。</td>
    </tr>
    <tr>
        <!-- 列的方向占据4个cell -->
        <td rowspan='1'><a href="./KernelLaunch"> KernelLaunch</td><td>Kernel Launch方式调用动态输入AddNCustom算子。</td>
    </tr>
</table>

## 算子描述
AddN算子实现了两个数据相加，返回相加结果的功能，其中核函数的输入参数为动态输入，动态输入参数包含两个入参，x和y。对应的数学表达式为：  
```
z = x + y
```
## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddN</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x（动态输入参数srcList[0]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y（动态输入参数srcList[1]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">addn_custom</td></tr>
</table>

## 支持的产品型号
- Atlas 推理系列产品（Ascend 310P处理器）
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 目录结构介绍
```
├── FrameworkLaunch         //使用框架调用的方式调用动态输入的AddN自定义算子工程。
└── KernelLaunch            //使用核函数直调的方式调用动态输入的AddN自定义算子。
```
## 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

## 编译运行样例算子

### 1. 准备：获取样例代码<a name="codeready"></a>

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```bash
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
   cd ${git_clone_path}
   git clone https://gitee.com/ascend/samples.git
   ```
   **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
   ```bash
   git checkout v0.5.0
   ```
 - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。
 
   **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。下载压缩包命名跟tag/branch相关，此处以master分支为例，下载的名字将会是samples-master.zip**
   ```bash
   # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。
   # 2. 将ZIP包上传到开发环境中的普通用户某个目录中，【例如：${git_clone_path}/samples-master.zip】。
   # 3. 开发环境中，执行以下命令，解压zip包。
   cd ${git_clone_path}
   unzip samples-master.zip
   ```
### 2. 编译运行样例工程
- 若使用框架调用的动态输入方式，编译运行操作请参见[FrameworkLaunch](./FrameworkLaunch)。
- 若使用核函数直调的动态输入方式，编译运行操作请参见[KernelLaunch](./KernelLaunch)。
## 更新说明
| 时间       | 更新事项                                            |
| ---------- | --------------------------------------------------- |
| 2024/10/01 | 增加动态输入样例                                     |