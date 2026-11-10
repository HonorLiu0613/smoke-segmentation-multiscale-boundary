# 烟雾图像语义分割复现实验

对应论文《融合多尺度特征与边界监督的烟雾图像语义分割方法研究》，作者刘浩然。

## 实验范围

这是独立实现的紧凑U形网络及多尺度、边界辅助监督消融，不是Trans-BVM、FoSp或原版大规模U-Net的复现。实验使用固定400幅训练、100幅验证、400幅测试图像，128×128输入，12轮训练，随机种子42和2026。无人机火灾巡检是论文应用背景；本次实验数据未被认定为无人机航拍数据。

## 文件

- `smoke_experiment.py`：网络、损失、分组划分、训练、评价、扰动实验与计时。
- `splits.json`：本次实验固定文件清单。复现时保留该文件。
- `download_data.py`：按固定Git提交下载公开数据镜像，核对版本并记录来源。
- `verify_reproduction.py`：检查共享初始化、有限梯度、边界指标、分组与图像字节重复。
- `infer.py`：加载已有权重，对单幅图像预测二值烟雾掩膜。
- `export_visuals.py`：导出真实区域概率、边界概率和前景面积分布，供科研图件使用。
- `results/`：8组真实运行的权重、逐轮日志、预测、汇总CSV和运行环境。

## 环境与运行

原实验使用Windows 11 x64（10.0.26200）、Python 3.12.4、PyTorch 2.13.0+cpu、NumPy 2.4.4、Pillow 12.2.0、Intel Core i5-13500H（12核16线程）、16 GiB内存，并将PyTorch限制为4个计算线程。训练和评价使用FP32，所用PyTorch构建没有CUDA后端。完整运行参数见`results/environment.json`。

在本目录打开终端，建议创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe verify_reproduction.py
.\.venv\Scripts\python.exe smoke_experiment.py --cpu
```

`results/`中已存在的同名运行会直接读取缓存。要重新训练并保留原结果，可执行`python smoke_experiment.py --cpu --output-dir results_retrained`。重新训练时应保留`splits.json`对应的固定划分。原始`results/`继续保留论文对应的权重与日志。

代码依赖的最低版本见`requirements.txt`；相同种子无法保证跨PyTorch版本和硬件逐比特一致，应同时核对环境、输入清单和训练配置。默认未使用外部预训练权重，也未根据测试结果调整阈值。

## 数据

原始研究：Yan S, Zhang J, Barnes N. Transmission-Guided Bayesian Generative Model for Smoke Segmentation. AAAI 2022. DOI: 10.1609/aaai.v36i3.20207。

相关Git仓库：

- 原始研究代码：[SiyuanYan1/Transmission-BVM](https://github.com/SiyuanYan1/Transmission-BVM)。
- FoSp参考方法：[LujianYao/FoSp](https://github.com/LujianYao/FoSp)。

这些链接指向数据与参考方法。本次课程实验的独立实现仓库为：https://github.com/HonorLiu0613/smoke-segmentation-multiscale-boundary 。

镜像：[MultiX-Amsterdam/smoke5k](https://github.com/MultiX-Amsterdam/smoke5k)，固定提交`844c7fe2153031240ebd7856b8054f09a58f6f86`。实际下载版有5058组训练文件、400组测试文件。按文件名启发式分组剔除108幅来源标识与测试重合的训练候选，再从其余4950幅中固定抽样。完整数据保留在本工作目录的`data/`，压缩复现包不包含整套数据，可由下载脚本获取。使用数据时请查阅原作者发布说明并引用原论文。

## 评价口径

- IoU、Dice、精确率、召回率：全测试集前景像素累计，不是前景和背景平均的mIoU。
- Boundary F1：3×3形态学边界带，128×128分辨率下1像素切比雪夫距离容差，边界匹配分别计算查准率与查全率。
- 亮度扰动：RGB乘以0.5；模糊扰动：反射填充的5×5均值滤波，两类扰动分别施加。
- 模型选择：最高验证IoU轮次；区域预测阈值0.5。
- 计时：CPU批量1，预热10次，完整前向50次中位数，包含边界分支，排除读盘与预处理。
- 表格：两个随机种子的均值±总体标准差，没有进行统计显著性检验。

## 单图推理

```powershell
python infer.py --checkpoint results/combined_42.pt --image data/smoke5k-main/test/img/1498417233_+00840.jpg --output prediction.png
```

输入文件名应以本地实际存在的图像为准。输出是二值掩膜，并通过最近邻插值恢复到原图尺寸；恢复尺寸不会增加128×128预测中缺失的细节。

## 结果解读

模型规模和训练预算很小，结果只用于本论文的受控比较。所有数字来自实际执行，具体结果见`results/summary.csv`，不能直接与训练数据、预训练设置、分辨率或评价口径不同的论文数字比较。现有结果未完成真实无人机视频、火源地理定位、机载功耗或端到端告警验证。

基础模型与联合模型的烟雾IoU分别为10.25±1.47%和11.52±2.55%，联合模型平均提高1.26个百分点；Boundary F1平均提高1.67个百分点。均值和总体标准差仅来自两个随机种子，不能据此宣称统计显著或SCI层面的原创突破。

## 论文图件

压缩包根目录的`figures/`提供4幅图的PNG和矢量PDF：前景面积分布、验证集IoU曲线、烟雾与边界概率热力图、逐图IoU分布与按烟雾面积分组的配对差值。绘图源代码为根目录的`tools/make_figures.py`。

图3使用种子42的真实最佳权重输出，样本按真实烟雾面积的10%、50%、90%、99%分位数选取。所有概率图使用统一0—1色标，前两行加入黄色ROI及局部放大。图4对每幅含烟测试图像先汇总两个种子的IoU，再绘图，样本数为399；唯一空掩膜图像只在该分布分析中排除，主结果表仍使用全部400幅测试图像。

已保存`results/research_visuals.npz`和`results/visualization_protocol.json`，可直接重绘，无需重新训练。在压缩包根目录执行：

```powershell
python -m pip install "matplotlib>=3.10"
python tools/make_figures.py
```

若需从权重重新导出概率，则在`code/`目录、下载数据后执行`python export_visuals.py`，再回到包根目录重绘。原绘图环境使用Matplotlib 3.10.7和Times New Roman字体。
