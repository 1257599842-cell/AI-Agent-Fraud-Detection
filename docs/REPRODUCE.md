# 分层复现指南

按所需资源选择入口。**浏览项目不要求先下载数据、训练模型或购买 LLM API。**

| 层级 | 需要什么 | 得到什么 |
|---|---|---|
| 0 · 看展示 | 浏览器 | 历史案例、可展开证据、成本沙盘 |
| 1 · 核对仓库 | Python；单元测试另需项目依赖 | 页面构建一致性、测试、归档哈希 |
| 2 · 重建模型与特征 | 自行取得 IEEE-CIS 数据、CPU、磁盘和时间 | 图特征、模型、评分缓存、知识库 |
| 3 · 本地 API | 完成层级 2 | 原始字段评分、已有交易调查、模板降级 |
| 可选 · 浏览器验收 | Playwright 与浏览器 | 真实渲染、移动端、公式、离线交互检查 |

## 0. 无凭证查看展示

打开 [GitHub Pages](https://1257599842-cell.github.io/AI-Agent-Fraud-Detection/)，或将仓库下载到本地后直接用浏览器打开 docs/index.html。

页面数据全部内联。断网时案卷、主题切换和沙盘仍可用；跳转到 GitHub 源报告需要网络。它不会调用评分服务或付费 LLM。

## 1. 核对代码和归档，不训练模型

在项目根目录执行：

~~~bash
python3 -m src.serving.build_demo_page --check
~~~

这一步只依赖 Python 标准库，只读检查模板、案例数据、两份 HTML 和 .nojekyll。修改模板后，运行以下命令更新派生产物：

~~~bash
python3 -m src.serving.build_demo_page
~~~

运行单元测试：

~~~bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -t .
python -m src.eval.report_manifest
python -m src.eval.modelcard_review
~~~

测试使用合成小样本、mock 和随仓库发布的归档，不要求下载原始交易数据，也不发送付费请求。报告清单检查核对登记的报告和冻结归档是否变化，**不等于审定实验结论正确**。

项目统一使用 **Python 3.13**；本地检查使用 macOS，CI 使用 Linux。Windows 尚未验证。macOS 的 LightGBM 如提示 OpenMP 缺失，需先安装相应系统运行库（如 `brew install libomp`）。

### 依赖分层与版本维护

| 清单 | 用途 |
|---|---|
| [requirements.txt](../requirements.txt) | 模型、API、报告与数据无关测试；不安装 JupyterLab 或浏览器 |
| [requirements/browser.txt](../requirements/browser.txt) | 核心环境 + Playwright；CI 使用此清单 |
| [requirements/notebooks.txt](../requirements/notebooks.txt) | 核心环境 + JupyterLab；只在交互分析时需要 |

三个清单由对应 `.in` 文件生成，直接和传递依赖均固定版本；扩展环境以核心清单为约束。可选的实时 LLM SDK 不在这些测试环境中，没有安装它也能执行上述检查。

需要交互 Notebook 时，运行 `python -m pip install -r requirements/notebooks.txt`。维护者更新版本时，先修改 [core.in](../requirements/core.in) 或对应扩展 `.in`，再在独立的 Python 3.13 维护环境安装 `pip-tools==7.6.1` 并运行：

~~~bash
python -m piptools compile --no-header --no-annotate --strip-extras \
  --output-file requirements.txt requirements/core.in
python -m piptools compile --no-header --no-annotate --strip-extras \
  --output-file requirements/browser.txt requirements/browser.in
python -m piptools compile --no-header --no-annotate --strip-extras \
  --output-file requirements/notebooks.txt requirements/notebooks.in
~~~

更新后须在干净虚拟环境安装，执行 `pip check`、完整测试及 Linux CI；浏览器版本变动还要重跑真实渲染检查。清单锁定 Python 包版本，不锁操作系统、OpenMP 或二进制构建；Windows 条件依赖不在已验证范围内，模型重训也不承诺跨平台逐字节一致。

## 2. 用自己的合规数据副本重建

先在 [IEEE-CIS 比赛页面](https://www.kaggle.com/competitions/ieee-fraud-detection)接受条款，下载并放置：

~~~text
data/
  train_transaction.csv
  train_identity.csv
~~~

官方无标签 test 文件不参与这里的线下评估。随后运行：

~~~bash
python -m src.features.load_data
python -m src.features.graph_features
python -m src.agent.disposition
python -m src.agent.knowledge
~~~

这会构建合并数据、21 天合成标签成熟期的图特征、表 + 图模型及评分缓存、规则与案例库。disposition 首次运行需要训练；分数和模型均存在时会复用缓存，缺少持久化模型文件则会重建。**这些命令会写本地数据、模型与部分报告，不要把它们当作只读检查。**

主要实验入口：

~~~bash
python -m src.model.graph_vs_tabular
python -m src.model.label_availability_audit
python -m src.features.build_duckdb
python -m src.eval.online_replay --run
~~~

它们会消耗 CPU、内存和磁盘，并可能更新对应报告。主对照与独立标签审计采用不同协议，应分别解释，不能把绝对分数差直接归因于单一因素。完整口径见 [实验协议](../EXPERIMENT_PROTOCOL.md)。

~~~bash
python -m src.eval.report_manifest --verify-rerun
~~~

这个可选命令还会重跑登记的低成本生成器：它可能依赖层级 2 的数据与缓存，且会改动派生缓存，**不是干净克隆下的零数据检查**。不要与其他报告构建并行执行。

## 3. 启动本地 API

完成上述资源构建后：

~~~bash
uvicorn src.serving.app:app --host 127.0.0.1 --port 8000
~~~

~~~bash
curl http://127.0.0.1:8000/healthz

curl -X POST http://127.0.0.1:8000/score \
  -H 'Content-Type: application/json' \
  -d '{"transaction_dt":16000000,"fields":{"TransactionAmt":100,"ProductCD":"W","card1":1234}}'
~~~

/score 必须收到有限、非负金额。示例仅提供少量原始字段，其余作为缺失值处理，用于接口演示，不代表完整业务输入的效果。

- /score 计算 27 列历史特征，15 列图特征进入模型，12 列 velocity 仅诊断返回；不调用 LLM。
- 服务预灌 day < 146 的历史快照，**请求不会自动追加历史**。独立回放脚本才执行“先查后写”。
- /investigate 接受已有评估交易的 ID，使用离线证据后端；尚未接通任意新交易的调查。
- 实时 LLM 是可选项，需要额外 SDK、可用模型和相应服务凭证；不提供凭证时可能闸门放行或模板降级。本文的复现步骤不要求开启付费调用。

容器演示（镜像不含数据与模型）：

~~~bash
docker build -t fraud-copilot .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" fraud-copilot
~~~

## 可选：检查网页真实渲染

~~~bash
python -m pip install -r requirements/browser.txt
python -m playwright install chromium
python -m src.serving.check_demo_page --browser chromium
python -m src.serving.shoot_demo --browser chromium --check-only
~~~

已有本机 Chrome 时，可省略浏览器下载，在后两个命令末尾加 --channel chrome。也支持安装 WebKit 后使用默认 --browser webkit；单一引擎通过不代表所有浏览器均通过。

- check_demo_page：字号、对比度、资源依赖、400 组前后端成本公式对账、案例引用。
- shoot_demo --check-only：5 种视口 × 深浅主题 × 7 案例，检查溢出、成本切换、负成本提示、降级、JS 错误及远程请求。
- 不加 --check-only 会更新 reports/demo/shots/ 中的三张当前展示截图。

发布模板位于 [reports/demo/_template.html](../reports/demo/_template.html)，事实数据来自 [demo_data.json](../reports/demo/demo_data.json)。构建器同时写出 reports/demo/index.html 与 docs/index.html；不要手工只改其中一份。历史案例原文和当前导读分开存放，不通过改写原始报告美化结果。
