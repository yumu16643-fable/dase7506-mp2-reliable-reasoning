# 算术一致性优化结果

本轮新增十八个可核验的算术特征，检查候选解答中的数字等式，并与原有表示/统计特征结合。模型选择仅用训练集内按题目划分的交叉验证；验证集在本轮选定模型后检查。本轮不读取公开测试输入或参考答案，不修改正式提交。

| 评估 | 原提交模型 | 本轮候选 |
|---|---:|---:|
| 训练集内三折（seed 2026） | 90.81%（4318/4755） | 91.65%（4358/4755） |
| 重新划分三折（seed 2037） | 90.75% | 91.84% |
| 验证集 | 92.23%（736/798） | 94.24%（752/798） |
| 公开测试集 | 已提交 81.37%（380/467） | 本轮未评估 |

在相同列表排序损失、正则化参数和折划分下，移除算术特征后三折成绩为 91.23%；加入后为 91.65%，说明新特征在这个对照下增加二十题。完整方法相对原提交模型增加四十题，其中还包含训练目标变化带来的影响。

重新划分三折是对同一训练集的诊断，不是新的独立数据集。当前验证集及训练折也在之前实验中使用过，因此不能把本轮结果视为完全独立的泛化证明。已有开发证据比前一轮更一致，但尚无新的公开测试成绩。

## 算术检查的范围

使用有界 AST 解释器计算纯数字表达式，不使用 Python eval，不执行函数、属性访问或任意代码。支持部分分数、小数、百分比、LaTeX 和乘除符号，允许合理的显示精度误差。不能可靠提取的代数表达式、单位相关式子等会被跳过。

约 63.57% 的训练候选至少有一个可解析数值等式。带有检查到的不一致的比例，在最终答案正确的训练候选中为 1.63%，错误候选中为 6.02%。中间步骤错误可能仍得到正确最终答案，解析也有边界，所以不直接用算术标记淘汰候选，而让训练模型决定其权重。

本轮采用 CPU 训练轻量线性排序器。选中的是带边际列表损失、L2=0.1 的模型，输入共 1088 维。全部必需推理资产合计 **14,323 字节**；本地完整 CPU 验证集预测加评分约 **1.896 秒**，不需要 PyTorch 或 GPU。该时间不是课程参考 Xeon 上的测量。

## 复现与记录

- `reports/arithmetic-report.pdf`：本轮四页报告，含方法、消融、复核、失败案例、资源和 AI 披露。
- `code/arithmetic_features.py`：数字表达式检查与十八个特征。
- `code/check_arithmetic.py`：分数、小数、LaTeX、舍入、单位和非法表达式行为测试。
- `code/train_arithmetic.py`：训练内三折选择和最终训练，不读取测试集。
- `code/check_robustness_arithmetic.py`：换 seed 2037 的三折诊断，不重新选择模型。
- `code/ablate_arithmetic.py`：相同训练目标/正则化下的特征消融。
- `code/submission_arithmetic.py`：NumPy 推理。
- `code/runs/model_bundle_arithmetic/`：固定权重和配置。
- `code/runs/freeze-arithmetic.json`：对应源代码和资产的 SHA-256 哈希。
- `code/runs/arithmetic-validation-score.json`：官方独立评分器输出，752/798。
- `code/runs/arithmetic-search/`：完整模型比较、复核、消融和解析审计。

在 `code/` 中安装 `requirements.txt`，执行 `python download_data.py` 获取官方数据，然后运行 `python check_arithmetic.py` 和 `python evaluate_arithmetic.py` 可复现冻结的验证结果，无需训练。

训练复现：安装 `requirements-training.txt`，运行 `python train_arithmetic.py`。完整训练/验证过程不会读取测试输入或测试参考答案。

本轮发现并修正了一个数值片段提取问题；初步调试运行在验证前中止，原始部分日志保存在 `arithmetic-parser-debug.log`。报告只使用修正后完整实验的结果。

原正式提交仍为：https://github.com/xudongwu-0/xudongwu-0.github.io/issues/340
