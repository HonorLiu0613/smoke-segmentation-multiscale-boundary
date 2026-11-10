# 融合多尺度特征与边界监督的烟雾图像语义分割

机器人工程课程研究的可复现代码，作者：刘浩然。项目独立实现紧凑U形分割网络，在固定数据划分、共享主干初始化和相同训练预算下，对多尺度特征与边界辅助监督进行四组消融。无人机火灾巡检是应用背景；本次测试集未被确认为无人机航拍数据。

仓库保留了实际运行的8组最佳权重、逐轮日志和测试预测。实验精度较低，结论限于当前原型与训练设置；两个随机种子的结果不构成统计显著性证据。

阅读代码与准备课程答辩可先查看[网络结构与代码讲解](CODE_GUIDE.md)，其中给出了逐层张量尺寸、多尺度模块、边界监督、训练过程及创新性定位。

## 快速开始

需要Git和Python 3.12。下列命令在仓库根目录执行，`python`应指向你为本项目准备的Python环境。

```bash
git clone https://github.com/HonorLiu0613/smoke-segmentation-multiscale-boundary.git
cd smoke-segmentation-multiscale-boundary
python -m pip install -r code/requirements.txt
python code/download_data.py
python code/verify_reproduction.py
```

下载脚本获取约215 MB的公开数据压缩文件，并核对固定Git提交。仓库本身不包含原始数据集。

## 已保存结果与重新训练

默认命令读取已保存的同名实验，不会重复训练：

```bash
python code/smoke_experiment.py --cpu
```

要从随机初始化重新完成4个配置×2个种子的训练，请使用新的结果目录：

```bash
python code/smoke_experiment.py --cpu --output-dir code/results_retrained
```

这会重新训练，不覆盖仓库中用于论文的原始结果。请保留`code/splits.json`；它固定了训练、验证和测试文件清单。新实验的权重、日志与指标将保存在`code/results_retrained`。后续预测时可把权重路径换成该目录下的文件。

## 单图预测

```bash
python code/infer.py --checkpoint code/results/combined_42.pt --image your_image.jpg --output prediction.png
```

把`your_image.jpg`替换为实际图片路径。输出是黑白烟雾掩膜；推理输入为128×128，输出会恢复至原图尺寸，恢复尺寸不会补回已经丢失的细节。

## 重新绘制论文图件

仓库已保存绘图所需的概率和统计数组，因此直接绘图无需下载数据或重新训练：

```bash
python -m pip install -r tools/requirements.txt
python tools/make_figures.py
```

输出位于`figures/`，包括PNG和矢量PDF。若需重新从最佳权重导出概率，在下载数据后运行：

```bash
python code/export_visuals.py
python tools/make_figures.py
```

热力图由种子42的已训练权重输出，经sigmoid映射后共享0—1色标。4个样本按真实烟雾面积的10%、50%、90%、99%分位数选择，不按预测效果挑选。前两行提供黄色ROI与局部放大。性能分布图先对每幅图像的两次运行IoU取均值，再以399幅含烟图像绘图；主结果表使用全部400幅测试图像。

## 实验配置

| 项目 | 实际设置 |
|---|---|
| 数据 | SMOKE5K公开镜像；训练400、验证100、测试400 |
| 输入与主干 | 128×128；通道8/16/32/64；GroupNorm；随机初始化 |
| 训练 | Adam；学习率0.001；批量16；12轮；种子42、2026 |
| 增强 | 水平翻转、0.8—1.2亮度缩放 |
| 损失 | 区域BCE+Dice；边界平衡BCE，权重0.2 |
| 模型选择 | 最高验证集烟雾IoU轮次；测试阈值0.5 |
| 硬件 | Intel Core i5-13500H；16 GiB RAM；CPU，4计算线程，FP32 |
| 软件 | Windows 11 x64 10.0.26200；Python 3.12.4；PyTorch 2.13.0+cpu；NumPy 2.4.4；Pillow 12.2.0 |

原实验没有使用GPU。`code/requirements.txt`列出兼容性下限，并不是严格锁定文件；其他软件版本或硬件上的训练不保证逐比特一致。原运行元数据保存在`code/results/environment.json`。

## 实测结果

数值为两个随机种子的均值±总体标准差，单位为%。IoU为全测试集累计的烟雾前景IoU，不是包含背景类别的mIoU。

| 配置 | 烟雾IoU | Dice | Boundary F1 |
|---|---:|---:|---:|
| B：紧凑U形基础网络 | 10.25±1.47 | 18.56±2.42 | 7.00±0.14 |
| B+M：多尺度模块 | 9.49±0.18 | 17.33±0.29 | 7.22±0.69 |
| B+E：边界辅助监督 | 10.58±3.08 | 18.99±5.05 | 8.78±0.52 |
| B+M+E：联合模型 | 11.52±2.55 | 20.56±4.10 | 8.67±1.31 |

联合模型相对基础网络平均提高1.26个IoU百分点和1.67个Boundary F1百分点。多尺度模块单独加入后IoU下降；边界F1最高的是B+E。绝对精度较低，不能据此宣称具备可靠现场巡检能力。

Boundary F1采用3×3形态学边界带，在128×128上允许1像素切比雪夫距离容差。扰动分别使用RGB×0.5与5×5均值模糊。CPU延迟以批量1、预热10次后50次完整前向的中位数计，包含辅助边界分支，排除读盘与预处理。完整结果见`code/results/summary.csv`和`all_results.json`。

## 数据来源与研究局限

- 数据镜像：https://github.com/MultiX-Amsterdam/smoke5k
- 固定提交：`844c7fe2153031240ebd7856b8054f09a58f6f86`
- 原研究与代码：Yan S, Zhang J, Barnes N. Transmission-Guided Bayesian Generative Model for Smoke Segmentation. AAAI 2022. DOI: [10.1609/aaai.v36i3.20207](https://doi.org/10.1609/aaai.v36i3.20207)；https://github.com/SiyuanYan1/Transmission-BVM
- FoSp参考方法：https://github.com/LujianYao/FoSp

镜像实际枚举5058组训练文件和400组测试文件。依据文件名启发式来源分组，剔除108幅来源标识与测试重合的训练候选，再固定抽取训练与验证子集。该启发式不能完整恢复真实火灾事件。训练、验证、测试平均烟雾面积占比分别为25.01%、1.16%、2.23%，存在明显分布差异。

本项目没有直接运行Trans-BVM或FoSp训练代码，也没有完成真实航拍视频、跨事件泛化、机载功耗及端到端告警评估。数据集及参考方法的使用条件以原作者发布说明为准。

## 文件结构

```text
code/
  smoke_experiment.py       网络、损失、训练、评价与计时
  download_data.py          固定版本数据下载
  verify_reproduction.py    初始化、梯度、指标与划分核查
  infer.py                 单图预测
  export_visuals.py         从真实权重导出科研绘图数组
  splits.json              固定训练/验证/测试清单
  data/provenance.json     数据来源与下载校验值
  results/                 8组权重、日志、预测与汇总
figures/                   4幅论文图的PNG/PDF及样本清单
tools/make_figures.py       科研图件绘制
references/                已核验参考文献信息
CITATION.cff               项目引用信息
```
