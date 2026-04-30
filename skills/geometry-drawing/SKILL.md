---
name: geometry-drawing
description: 把数学题里的几何描述转换成 GeoGebra 2D / 3D 或 TikZ 代码，并通过画板 API 渲染。当用户要"画图""作图""把这道几何题画出来""生成 TikZ"时使用。
---

# 几何画板生成（GeoGebra + TikZ）

## 何时触发
- 题目含有「平面几何 / 立体几何 / 解析几何 / 函数图像」并且需要**可视化**
- 用户说："帮我画一下"、"出个图"、"GeoGebra 代码"、"TikZ 代码"

## 引擎选择规则
| 题目特征 | 选哪个 |
|---|---|
| 三角形、圆、抛物线、向量、解析几何 | **GeoGebra 2D**（默认） |
| 球、圆柱、圆锥、空间四面体、二元函数曲面 | **GeoGebra 3D** |
| 论文 / PDF / LaTeX 风格输出 | **TikZ** |

判断方式：如果题目里出现 `球|圆柱|圆锥|四面体|曲面|f(x,y)|空间` → 3D；用户明确说"TikZ" → TikZ；否则 2D。

## 调用流程

### 第 1 步：调后端 AI 接口
```
POST /api/academy/draw-ai
{
  "board_type": "2d" | "3d" | "tikz",
  "prompt": "画出题目描述的几何图形",
  "context": "<可选：已有的代码，在此基础上修改>",
  "question_content": "<题目原文，必填，让 AI 知道画什么>",
  "question_answer": "<参考答案，可选，帮 AI 画准>"
}
```
返回 `{ code, explanation }`，`code` 即可直接执行的画板代码。

### 第 2 步：渲染
- **GeoGebra**：在前端把 `code` 按行喂给 `applet.evalCommand(line)`，先 `newConstruction()` 清空。
- **TikZ**：调 `POST /api/render-tikz {code}`，拿到 `{ok, url}`，用 `<img src=url>` 显示。

### 第 3 步：保存到题目
```
PUT /api/academy/exams/{exam_id}/questions/{q_id}
{ "geogebra_code": "<code>", "viz_engine": "2d"|"3d"|"tikz" }
```

## 输出约定
- 代码与文字说明用 `---说明---` 分隔
- 不要使用 ```` ``` ```` 围栏，直接输出纯代码
- TikZ 必须包成 `\begin{tikzpicture} ... \end{tikzpicture}`
- GeoGebra 一行一条命令，注释用 `#` 开头

## 常见坑
- GeoGebra 中文标签会被解析成变量名 → 用英文字母命名点（A、B、C）
- TikZ 不支持中文 → 需要 `\usepackage{ctex}`，但服务端 `/api/render-tikz` 已经处理
- 立体几何用 2D 模式渲染会报错 → 一定要先选 3D
