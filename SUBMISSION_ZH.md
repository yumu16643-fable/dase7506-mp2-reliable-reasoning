# 提交说明

本文件为首次提交的历史说明。当前优化版的最终测试成绩为 82.44111349036403%，方法为 submission_arithmetic.py；请以根目录 README 和 FINAL_ARITHMETIC_ZH.md 为准。

本目录包含完整代码、实验和报告。正式提交是否完成，以课程网站或实际 GitHub 提交 issue 为准。

## 表单填写内容

- Student ID：**3036843484**
- Accuracy：**81.37044967880085**（填写百分数数值，不填 0.8137）
- Code link：发布后使用作业仓库的固定 commit 链接；该版本应包含本报告和匹配代码。
- Checkpoint / configuration link：同一 commit 下 `code/runs/model_bundle/` 文件夹的公开链接。
- 提交网站：https://xudongwu-0.github.io/courses/dase7506/mp2/
- 截止：2026 年 10 月 31 日 23:59，北京时间。

网页准备好内容后，必须点击 Submit on GitHub，并真正创建 issue。仅填写网页或生成链接不算提交。记下创建后的 issue 地址并检查榜单。

## 推荐发布方式

使用小型 publication 代码包作为 GitHub 仓库内容，不必再次上传官方大数据。仓库包含固定上游版本的数据下载脚本；评阅者运行 `python download_data.py` 后，官方哈希检查通过，即可离线评估。

仓库至少保留 `README.md`、`reports/report.pdf`、全部 `code/*.py`、固定依赖文件、`code/runs/model_bundle/`、`code/runs/freeze.json`、实验记录和官方评分 JSON。推荐直接发布完整的小型 publication 包内容，避免遗漏。

独立 checkpoint 包解压后，将其中的 `model_bundle` 放到 `code/runs/` 即可。完整 self-contained 包已经包含官方数据，可以直接按 README 复现。

## 提交前需要理解的内容

1. 这个方法从四个现成解答中选择一个，不生成新解答。评分只看最终数字，不验证整条推理。
2. 使用 1024 维官方表示、1024 维同组中心化表示，以及 46 个文本/答案统计特征；训练集拟合标准化和逻辑回归，验证集选择 C=0.01。
3. 验证集 736/798=92.23%；方案冻结后测试集 380/467=81.37%。测试集全部包含答案分歧，验证集有很多简单的一致答案组，因此不能直接把分差都解释成过拟合。
4. 模型资产共 26,397 字节；本地完整测试预测加评分约 0.925 秒。课程参考服务器的时间仍由课程方判断。
5. 逻辑回归中中心化的组均值项在同题候选比较时会抵消；它改变拟合/正则化，不能声称因此获得复杂的跨候选推理能力。
6. 模型仍会漏过算术错误、配对数量关系错误和题意歧义。报告包含实际失败案例。
7. 已如实披露 Codex 对方案、代码、实验、分析和报告的实质性帮助。请在提交前阅读并理解。

## 发布目标

GitHub 账号：`yumu16643-fable`。

仓库：https://github.com/yumu16643-fable/dase7506-mp2-reliable-reasoning

检查点目录：https://github.com/yumu16643-fable/dase7506-mp2-reliable-reasoning/tree/main/code/runs/model_bundle

正式表单使用匹配代码版本的固定 commit 链接，避免后续分支更新影响复现。完成发布并实际创建课程提交 issue 后，才算正式提交。
