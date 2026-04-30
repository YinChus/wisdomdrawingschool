# 🏭 AI 出题工厂 — 多智能体「一题生 N 题」详细设计

> 教师在任意一道题旁点 **"AI 出变式"**，系统会基于这道**原题**自动产出 N 道**考点一致、形式不同**的新题（含题干、答案、解析、可选几何图）。教师勾选满意的，一键追加到本试卷末尾。

---

## 一、为什么不是单 Prompt？

| 单 prompt 方案 | 多 Agent 方案 |
|---|---|
| 一次性让 LLM "出题 + 解 + 审" | 每个职责一个 Agent，单独优化 prompt |
| 出错只能整道重来 | 错在哪个节点就重跑哪个节点 |
| 不可并行 | N 道变式题可**并行生成**，单题失败不影响其他 |
| 难度策略由模型自由发挥，质量飘忽 | **Planner** 显式规划 4 类策略，可控 |
| 模型自己判好坏，几乎不会说"不" | **Critic** 独立审稿，不通过则**自反馈循环**最多 2 轮再写 |

---

## 二、整体架构

```
                       原题（题干 + 答案 + 解析 + 几何代码）
                                  │
                                  ▼
                       ┌──────────────────┐
                       │ ① Planner 规划师 │   规划要出几道、每道改哪个维度
                       │  (出 N 个策略)   │   策略：换数 / 换情境 / 换图 / 加难
                       └─────────┬────────┘
                                 │
            ┌────────────┬───────┴───────┬────────────┐
            ▼            ▼               ▼            ▼          (并行 asyncio.gather)
       ┌────────┐   ┌────────┐      ┌────────┐   ┌────────┐
       │② Gen 1 │   │② Gen 2 │ …    │② Gen N │   │② Gen N │     生成题干
       └───┬────┘   └───┬────┘      └───┬────┘   └───┬────┘
           ▼            ▼               ▼            ▼
       ┌────────┐   ┌────────┐      ┌────────┐   ┌────────┐
       │③ Solver│   │③ Solver│ …    │③ Solver│   │③ Solver│     解出答案 + 解析
       └───┬────┘   └───┬────┘      └───┬────┘   └───┬────┘
           ▼            ▼               ▼            ▼
       ┌──────────────────────────────────────────────────┐
       │  ④ GeoDrawer 几何代码师                           │
       │  (按题型自动判 2D / 3D / 不需要画图)              │
       └─────────────────────┬────────────────────────────┘
                             ▼
                       ┌──────────────┐
                       │ ⑤ Critic 审稿│  自洽？答案唯一？难度对？
                       └─┬──────────┬─┘
                         通过        不通过 → 回到 ② Generator 重写（≤2 轮）
                         ▼
                       ┌──────────────┐
                       │ ⑥ Finalize   │  排序 + SSE 流式推给前端供教师勾选
                       └──────────────┘
```

**编排框架**：基于 **LangGraph 有状态图（StateGraph）**；全局 `MAX_FIX_ROUNDS=2`、`MAX_RECURSION=25` **双层熔断** 防止无限循环。

---

## 三、状态定义（FactoryState）

```python
class FactoryState(TypedDict, total=False):
    # 输入
    src_question: dict      # 原题：content / answer / solution / geogebra_code / viz_engine / category
    n: int                  # 期望生成数量
    # 中间
    plan: list[dict]        # [{strategy, focus, difficulty}]
    variants: list[dict]    # [{idx, content, answer, solution, geogebra_code, viz_engine,
                            #   critic, fix_count, status}]
    # 输出
    notes: list[str]        # 执行轨迹（SSE 流式推送给前端）
    error: str
```

> **设计要点**：所有 Agent 之间只通过这个**共享 State**通信，每个节点只读自己关心的字段、写自己负责的字段，强约束接口 + 弱耦合实现，任意 Agent 都可独立替换为更强模型。

---

## 四、六个 Agent 实现细节

### ① Planner ｜ 规划师

**职责**：读原题，规划要出几道变式、每道改哪个维度。

**输入**：原题题干 + 答案 + 解析摘要（≤ 400 字）
**输出**：长度为 N 的策略清单
```json
{
  "考点": "圆与圆柱的几何关系",
  "可视化需求": "yes",
  "plan": [
    {"strategy": "numbers", "focus": "把直径 2 换成直径 3", "difficulty": "easy"},
    {"strategy": "context", "focus": "把"球面 + 圆柱"换成"圆台 + 内接球"", "difficulty": "medium"},
    {"strategy": "extend",  "focus": "在原问基础上加(2)问求体积比", "difficulty": "hard"}
  ]
}
```

**强约束策略池（仅 4 类）**：

| 策略 | 含义 | 难度 |
|---|---|---|
| `numbers` | 只换数字 | easy |
| `context` | 换情境 / 包装（小明买苹果 → 工厂生产零件） | medium |
| `figure` | 换图形结构（仅几何题：三角形 → 四边形） | medium-hard |
| `extend` | 加问 / 加难度 | hard |

> **价值**：让 LLM 的"创造力"被显式约束在 4 个**可解释的维度**上，难度可调、变式策略可解释，规避单 Prompt 自由生成的质量飘忽问题。

---

### ② Generator ｜ 出题师 × N（并行）

**职责**：按 Planner 给的策略写新题题干。

**关键实现**：
```python
results = await asyncio.gather(
    *[_gen_one(src, p) for p in plan],
    return_exceptions=True,   # 单题异常不拖垮整批
)
```

**输出契约**（强 JSON Schema）：
```json
{
  "content": "新题目正文（含已知条件和问题）",
  "options": ["A. ...", "B. ...", "C. ...", "D. ..."]   // 仅选择题，否则 []
}
```

**Prompt 内嵌 LaTeX 书写规范**（从源头杜绝乱码）：
- 行内公式 `$..$`，独立公式 `$$..$$`
- 文字用 `\text{...}`，禁止 `\ext{...}`
- 反斜杠为单反斜杠，禁止双反斜杠
- 集合写作 `\{x \mid -1 \le x \le 1\}`

> **价值**：N 道变式无依赖关系 → 并行后总耗时 ≈ 单题耗时 + 调度开销，3 题不再线性放大成 3 倍。

---

### ③ Solver ｜ 解题师

**职责**：单独负责"做题"，与"出题"职责分离。

**输出**：
```json
{
  "answer":   "最终答案（简洁）",
  "solution": "第一步...第二步...∴ 结论"
}
```

**关键约定**：JSON 字符串里的 LaTeX 命令必须**双反斜杠**（避免 `\b\t\n\f` 被 JSON 转义成控制字符），并配套实现了 `_fix_latex` 兜底还原层（把退格/制表符/换页符还原回 `\b \t \f`）。

> **价值**：出题专家 ≠ 解题专家。让 Generator 专注"巧妙变式"，Solver 专注"严密推导"，分而治之提升整体质量。

---

### ④ GeoDrawer ｜ 几何代码师（按需触发）

**职责**：判断是否需要画图、走 2D 还是 3D 引擎、生成可执行代码。

**引擎自动路由**（关键词正则嗅探）：

| 命中关键词 | 选用引擎 |
|---|---|
| `三角形 / 椭圆 / 双曲线 / 抛物线 / 切线 / 焦点` | **GeoGebra 2D** |
| `四面体 / 棱柱 / 棱锥 / 球面 / 圆锥 / 圆柱 / 二面角 / 空间几何` | **GeoGebra 3D** |
| 都没命中 | **跳过画图**（代数题省 token） |

**输出**：可直接 `evalCommand` 执行的命令序列，前端画板即时渲染。

> **价值**：把"是否需要画图、画 2D 还是 3D"这件**强领域逻辑**用规则解决，不浪费 LLM 调用，也不会因模型一时糊涂给代数题硬塞图形。

---

### ⑤ Critic ｜ 审稿人 + Self-Refine 自反馈循环

**职责**：作为质量门，对【题干 + 答案 + 解析】三件套打分 + 提改进意见。

**输出**：
```json
{
  "ok": false,
  "score": 6,
  "issues": ["条件不足，无法唯一确定圆柱半径", "解析中第 3 步与答案矛盾"],
  "fix_hint": "在题干中补充'圆柱底面半径 = 球半径的一半'这一约束"
}
```

**自反馈循环（Self-Refine）**：
```
Critic 不通过
  ↓
带 fix_hint 回流到 Generator
  ↓
fix_count += 1
  ↓
重新生成 → 重新解题 → 重新审稿
  ↓
通过 OR fix_count >= MAX_FIX_ROUNDS (= 2) → 退出
```

**常见拦截**：条件不足 / 答案不唯一 / 跟原题雷同 / 难度与标签不符 / 几何题缺图形描述

> **理论依据**：思想直接来自 **Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., NeurIPS 2023)**。论文容易陷入"无限自我怀疑"，工程上加 `fix_count ≤ 2` 的硬上限避免该问题。

---

### ⑥ Finalize ｜ 装配 + 推流

**职责**：按难度排序、统一字段、把执行轨迹与最终结果通过 **SSE 流式推送**给前端。

**前端看到的实时轨迹**：
```
📋 Planner: 考点=圆柱与球，规划 3 道变式
🛠️ Generator: 并行生成 3 道变式题…
  ✓ 变式1 (numbers / easy): 已知圆柱的高为 2，它的两个底面…
  ✓ 变式2 (context / medium): 工厂生产一种圆柱形罐头…
  ✓ 变式3 (extend / hard): 在原问基础上求体积比…
🧮 Solver: 并行解答中…
🎨 GeoDrawer: 检测到 3D 题型，生成 GeoGebra 3D 命令…
🔍 Critic: 变式2 不通过 (5/10)，进入第 1 轮重写
  ↻ Re-Generator: 应用 fix_hint…
  ✓ 变式2 第 2 轮通过 (8/10)
✅ 全部完成，可勾选入库
```

教师在 UI 上勾选 → 调用入库接口 → 选中变式追加到本试卷末尾，自动复用原题的 `category` / `knowledge_points`。

---

## 五、防御性工程

| 风险 | 兜底措施 |
|---|---|
| LLM 输出非 JSON | 先去除 ```` ```json ```` 围栏，再正则抓第一个 `{...}` 块 |
| LaTeX 反斜杠被 JSON 转义成控制字符 | `_fix_latex` 把 `\b\t\f` 还原为 `\\b \\t \\f` |
| Critic 永不通过 | `MAX_FIX_ROUNDS = 2` 硬上限 |
| LangGraph 节点循环失控 | `MAX_RECURSION = 25` 全局兜底 |
| 单题 LLM 异常 | `asyncio.gather(return_exceptions=True)` 不拖垮整批 |
| 不需要画图却硬画 | 引擎路由先经过关键词嗅探，无几何关键词直接跳过 GeoDrawer |

---

## 六、核心收益

1. **可观测**：每步执行轨迹 SSE 推送，教师 / 开发都能实时看到「卡在哪个 Agent / 第几轮重写」
2. **可控**：Planner 显式策略池让难度与变化方向可解释、可调节
3. **可扩展**：所有 Agent 共享同一 State，强 Schema + 弱耦合，单点替换更强模型零成本
4. **稳定**：双层熔断 + 单题异常隔离，生产环境不会因为某次 LLM 抽风导致服务挂掉
5. **省钱**：Generator 并行 + GeoDrawer 按需触发，3 题 token ≈ 单题 × 1.1（vs. 串行 × 3）

---

## 参考文献

- **Self-Refine: Iterative Refinement with Self-Feedback** — Madaan et al., NeurIPS 2023
- **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** — Wu et al., Microsoft 2023
- **Plan-and-Solve Prompting** — Wang et al., ACL 2023
- **LangGraph: Building stateful, multi-actor applications with LLMs** — LangChain 官方文档

```

---

## 📊 实测性能（2026-04 实测）

> 用真实试卷数据跑出来的端到端基准，可重复执行：[`_bench_factory.py`](../_bench_factory.py)，详细报告：[`_bench_factory_report.json`](../_bench_factory_report.json)。

### 出题工厂：一题生 3 道变式（立体几何题：球内接圆柱）

| 节点 | 真实耗时 | 说明 |
|---|---|---|
| Planner 规划师 | **6.17 s** | 规划 3 道变式策略 |
| Generator ×3（**asyncio.gather 并行**） | **16.78 s** | 含 Self-Refine 容错机制 |
| Solver ×3（并行） | **1.62 s** | 解出三件套答案/解析 |
| GeoDrawer 几何代码师 | **34.06 s** | 立体几何 → 3D GeoGebra 代码 |
| Critic 审稿人 | **0.00 s** | 首轮全过，未触发自反馈 |
| **端到端总耗时** | **58.6 s** | 3 道完整变式 |
| **平均每道变式** | **~19.5 s** | 含规划/出题/解题/配图/审稿 |

> **token 消耗**：原题 52 → 输出 1146 tokens（含 3 套题干 + 答案 + 解析 + GeoGebra 代码）。

### GeoGebra 代码生成：按题型实测 3 类

| 题型 | 总耗时 | LLM 耗时 | 命令数 | 步骤快照 | 输出 token |
|---|---|---|---|---|---|
| 立体几何（Q8 球+圆柱） | **20.9 s** | 11.2 s | 12 条 | 9 步 | 261 |
| 函数（Q3 三次根+对数） | **2.49 s** | 2.49 s | 2 条 | 2 步 | 54 |
| 集合逻辑（Q1 区间交集） | **6.82 s** | 6.82 s | 9 条 | 5 步 | 146 |
| **平均** | **~10 s** | **6.85 s** | 7.7 条 | 5.3 步 | 154 |

**关键观察**：

- **MCP 工具层零开销**：`mcp_generate_geogebra_prompt` 构造提示词 < 1 ms，纯本地组装。
- **步骤快照解析零开销**：`_parse_snapshots` 基于正则解析 "# 步骤 N: 描述"，< 1 ms。
- 端到端时间几乎全部花在 LLM 推理上，工程层（注册表 + Schema + 解析）不构成瓶颈。

### 解析流水线：试卷文件 → 12 道结构化题（旁证）

| 阶段 | 耗时 | 数据 |
|---|---|---|
| PyMuPDF 文本抽取（4 页 PDF） | 0.10 s | — |
| LLM 结构化解析 | 11–22 s | 12 道题 |
| **端到端** | **~22 s** | **2705 input tokens** |

**Token 节省**：

- vs 朴素 base64 PDF 上传：**节省 99.5%**（530 561 → 2 705 tokens）
- vs Vision API 直接读图：**节省 38.8%**（4 420 → 2 705 tokens）

详细报告：[`_bench_report.json`](../_bench_report.json)
