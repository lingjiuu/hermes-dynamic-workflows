# Hermes Dynamic Workflows — 技术文档

模型现写一段受限 Python 脚本，后台运行时执行它，用 `agent()/parallel()/pipeline()`
编排大量独立 Hermes 子代理。本文逐点说明实现；所有结果字符串取自源码。

## 核心链路

主代理调 `workflow` 工具 → `WorkflowRunManager.start_from_params`：

1. 解析来源（`script` / `scriptPath` / `name` 三选一）。
2. `parse_script` + `extract_meta`：AST 校验，取首句字面量 `meta`。
3. 顶层启动审批（`require_launch_approval`，默认开）。
4. 持久化脚本、建 run 记录、起后台守护线程，**工具同步返回** Task ID / Run ID / 脚本路径。

后台线程 `run_workflow`：把 `meta` 之后的脚本体包成私有 async 入口，注入受限全局，
`exec` 执行。脚本里的 `await agent(...)` → `WorkflowAPI` → 并发槽 →
`HermesChildAgentRunner` 起一个独立 `AIAgent` 子代理，返回其文本（或 schema 校验后的对象）。

每次状态变化：快照写 run 记录 + `journal.jsonl`，子代理 transcript 实时导出。
终态：写 output 文件、最终 flush transcript、向主对话注入 `<task-notification>`（仅 CLI）。

落盘位置（`<cwd>` 为清洗后的工作目录）：

```
~/.hermes/projects/<cwd>/<sessionId>/workflows/scripts/<name>-<runId>.py   # 持久脚本
~/.hermes/projects/<cwd>/<sessionId>/subagents/workflows/<runId>/          # transcript 目录
    journal.jsonl                                                         # 运行事件流
    agent-<sessionId>.jsonl  +  .meta.json                                # 每个子代理
```

## Python Script API

脚本体本身即 async：直接写顶层 `await` / `return`，**首句必须是纯字面量
`meta = {...}`**（必填 `name`、`description`；可选 `whenToUse`、`phases`）。

| 全局 | 签名 | 说明 |
|---|---|---|
| `agent` | `await agent(prompt, opts=None)` | 起一个子代理。无 schema 返回文本；带 `schema` 返回校验后的对象。`opts`：`label` `phase` `schema` `model` `isolation` `agentType`。被用户跳过返回 `None`。 |
| `pipeline` | `await pipeline(items, stage1, …)` | 每个 item 独立流过各阶段，**无栅栏**。stage 回调收 `(prev, original, index)`；stage 抛错 → 该 item 变 `None`。多阶段默认用它。 |
| `parallel` | `await parallel(thunks)` | 并发执行，**栅栏**：全部完成才返回。单个失败 → 结果里为 `None`（整体不抛）。 |
| `phase` | `phase(title)` | 开启进度分组。 |
| `log` | `log(message)` | 向用户发一行进度。 |
| `workflow` | `await workflow(name_or_ref, args=None)` | 内联跑另一个工作流，共享并发/计数/停止/预算；仅一层嵌套。 |
| `args` | — | 工具入参 `args` 原样；未传为 `None`。 |
| `budget` | `budget.total` / `spent()` / `remaining()` | 取自用户消息里的 `+500k` 类目标。`total` 是硬上限，达到后 `agent()` 抛错；未设时 `remaining()` 为 `math.inf`。 |

其余可用：`json`、`math`、安全内建、常见异常类型。**禁止**（沙箱拒绝）：import、
文件/进程/网络、dunder 穿透、`eval/exec`、class 定义、动态调用目标、时间/随机 API
（破坏 resume）。

## 工具

插件向 Hermes 注册两个主代理工具（`workflow`、`task_stop`）和一个子代理专用工具
（`structured_output`，仅在带 schema 的子代理存活期间临时注册）。

### workflow

后台执行一段编排脚本；同步返回，完成时注入 `<task-notification>`。

工具 schema（描述极长，此处以 `…` 略去；参数完整如下）：

```json
{
  "description": "Execute a workflow script that orchestrates multiple subagents …",
  "parameters": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": false,
    "properties": {
      "script":          { "type": "string", "maxLength": 524288, "description": "Self-contained workflow script. Must begin with literal `meta = {…}` …" },
      "scriptPath":      { "type": "string", "description": "Path to a workflow script on disk. Takes precedence over `script` and `name`." },
      "name":            { "type": "string", "description": "Name of a predefined workflow (built-in or from .hermes/workflows/)." },
      "args":            { "description": "Value exposed to the script as global `args`, verbatim. Pass real JSON, not a JSON-encoded string." },
      "resumeFromRunId": { "type": "string", "pattern": "^wf_[a-z0-9-]{6,}$", "description": "Run ID to resume from; unchanged agent() calls return cached results." },
      "description":     { "type": "string", "description": "Ignored — set it in the script's meta block." },
      "title":           { "type": "string", "description": "Ignored — set it in the script's meta block." }
    }
  }
}
```

启动时还会把当前可用的 agentType 列表追加到描述末尾。

**工具调用结果。** 启动是同步的——下列结果在后台线程开始前返回，此时**没有**
notification（run 尚未开始）。

成功启动：

```
Workflow launched in background. Task ID: <taskId>
Summary: <meta.description or meta.name>
Transcript dir: <…/subagents/workflows/<runId>>
Script file: <…/scripts/<name>-<runId>.py>
Run ID: <runId>
To resume after editing the script: Workflow({scriptPath: "<path>", resumeFromRunId: "<runId>"})
You will be notified when it completes. Use /workflows to watch live progress.
```

校验/解析/输入类错误统一包成 `{"error":"<msg>"}`（`<msg>` 为下列之一）：

```
# 来源缺失
provide one of script, scriptPath, or name

# meta 契约（首句必须是纯字面量 meta 字典）
Invalid workflow script: `meta = {...}` must be the FIRST statement in the script
Invalid workflow script: meta must be a pure literal
Invalid workflow script: meta must be a pure literal: only plain properties allowed in meta
Invalid workflow script: meta must be a pure literal: template interpolation not allowed in meta
Invalid workflow script: meta must be a pure literal: non-literal node type in meta: <NodeType>
Invalid workflow script: meta.name must be a non-empty string
Invalid workflow script: meta.description must be a non-empty string
Invalid workflow script: meta keys must be strings
Invalid workflow script: forbidden meta key: <key>
Invalid workflow script: meta.<name|description|whenToUse> must be a string
Invalid workflow script: meta.phases must be a list
Invalid workflow script: meta.phases object entries require a title string
Invalid workflow script: meta.phases.<detail|model> must be a string
Invalid workflow script: meta.phases entries must be strings or objects

# 解析 / 体量
Invalid workflow script: Script parse error: <syntax msg> at line <l>, column <c>. Workflow scripts must be plain Python.
Invalid workflow script: workflow script is too large (<n> chars; max <max>)
do not define workflow(); the workflow script body is already async

# 沙箱（能力越界）
forbidden Python syntax: <NodeType>
forbidden name: <name>
forbidden attribute access: <attr>
forbidden method call: <attr>
dynamic call targets are not allowed
workflow script is too complex (>2500 AST nodes)
string literal is too large
integer literal is too large
bare 'except:' is not allowed; catch Exception or a specific type
'except BaseException' is not allowed; catch Exception instead
Workflow scripts must be deterministic: current time and randomness are unavailable (breaks resume). Stamp results after the workflow returns, or pass timestamps via args.

# resume 目标仍在运行
Workflow <runId> is still running (task <taskId>). Stop it first with task_stop({"task_id":"<taskId>"}) before resuming.
```

启动审批未通过同样走 `tool_error`（干净，无 trace）：

```json
{"error":"Workflow \"<name>\" was not launched: <reason>. Do not retry; tell the user it needs their approval."}
```

`<reason>` 取自审批环节，常见：`workflow launch was denied`、`workflow launch was
denied or timed out`、`launch approval required but no interactive channel (…)`、
`launch approval required but Hermes' approval engine is unavailable`。

其它**未预期**异常（真正的内部错误）才返回带 trace 的诊断信息——`trace` 是
`traceback.format_exc` 的最后 8 行（文件路径、行号、出错代码），随工具结果一并对模型
可见，便于报告：`{"error":"<Type>: <msg>","trace":"<最后 8 行回溯>"}`。

**完成通知。** run 进入终态后（仅 CLI）注入：

```
<task-notification>
<task-id><taskId></task-id>
<output-file><path></output-file>        # 有 output 文件时
<status><completed|failed|stopped|…></status>
<summary>Dynamic workflow "<name>" <completed | was stopped | failed: <error> | <status>: <error>></summary>
<result><结果，超过 notify_result_preview_chars 截断并提示 full result in <file>></result>   # 无 error 时
<recovery>Agent transcripts: <transcriptDir></recovery>      # 有 error 时
<usage><agent_count>N</agent_count><subagent_tokens>T</subagent_tokens><tool_uses>U</tool_uses><duration_ms>D</duration_ms></usage>
</task-notification>
```

### task_stop

按 Task ID 停一个后台 run（只作用于存活中的 run；已完成/历史 run 视为找不到）。

工具 schema：

```json
{
  "description": "- Stop a running workflow by its Task ID …",
  "parameters": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": false,
    "properties": { "task_id": { "type": "string", "description": "The ID of the workflow task to stop" } },
    "required": ["task_id"]
  }
}
```

工具调用结果：

```jsonc
// 成功（紧凑 JSON）
{"message":"Successfully stopped task: <taskId> (<summary>)","task_id":"<taskId>","task_type":"local_workflow"}

// 缺参数
{"error":"Missing required parameter: task_id"}

// 找不到 / 非存活状态（非 queued|running|paused）
{"error":"No task found with ID: <taskId>"}
```

### structured_output（子代理专用）

仅在带 `schema` 的子代理存活期间临时注册；该子代理工具的参数被替换成 `agent(…,
{"schema"})` 请求的 schema，子代理必须调它来提交最终结果。`agent()` 返回校验后的对象，
无需解析子代理文本。最多 `MAX_STRUCTURED_OUTPUT_RETRIES = 5` 次尝试。

子代理若不提交就想结束，会被追加一句继续指令、留在同一会话：

```
You MUST call the structured_output tool to complete this request. Call this tool now.
```

工具调用结果：

```jsonc
// 成功（纯文本，非 JSON）
Structured output provided successfully

// 校验失败：errors 为各项以 ", " 连接
{"error":"Output does not match required schema: <errors>"}
//   单项形如： /path/to/field: must have required property 'x'
//             /path: must be <type>      root: must NOT have additional properties
//             /path: must be equal to one of the allowed values     /path: must match pattern "…"

// 未注册期望（理论上不该出现）
{"error":"Output does not match required schema: root: no structured-output expectation is registered for this task"}

// 超过最大尝试次数
{"error":"Output does not match required schema: root: maximum structured output attempts exceeded (5)"}
```

> 校验优先用 `jsonschema`（Draft 2020-12）；未安装时退化为内置简易校验器（覆盖
> object/array/string/number、required、enum、additionalProperties 等常见关键字）。

## Prompt Cache

子代理是独立 `AIAgent`，继承 Hermes 的 prompt caching：对可缓存模型（Claude 系、
DashScope Qwen…）注入 `cache_control` 断点，每个子代理在自己的多轮工具调用间复用
`[tools + system]` 前缀。

**跨子代理共享前缀**：Hermes 的 `system_and_3` 策略把断点放在 system 末尾。为此
**子代理 system 提示在整个 fan-out 中保持逐字节一致**——只含稳定脚手架（基础指令 +
agentType 指令），而每任务上下文（工作区、label、phase、worktree 提示）放进子代理
首条 user 消息（`build_child_task_message`）。同 toolset + 同 agentType 的子代理因此
共享缓存前缀。每子代理的 `cache_read`/`cache_write` 在 `/workflows <runId> agent <id>`
可见；省幅取决于 provider（非可缓存模型为 0）。

## 并发与限额

- **并发槽**：每 run 一个信号量，上限 `concurrency`（默认 `min(16, cpu-2)`，且 ≤
  `max_concurrency`=16）。`parallel()/pipeline()` 可投递任意多项，同时只跑约槽数个，
  其余排队。
- **agent 上限** `max_agents`（默认 1000）：失控回退闸，远高于任何真实工作流。
- **循环闸**：每次 `while/for` 迭代注入 `__wf_tick__()`，检查停止 / deadline / 循环
  上限 `max_loop_iterations`（默认 1e7）。让 deadline 能在纯计算循环里触发。
- **deadline** `workflow_timeout_seconds`（默认 900s，暂停期间不计）。
- **子代理超时** `child_timeout_seconds`（默认 300s）：单个超时记 `WorkflowTimeout`
  抛回脚本（可 `try/except`）。

run 级硬停（用户停止、deadline、预算/agent/循环上限）派生自 `BaseException`，脚本的
`except Exception` 吞不掉；沙箱并禁止 `except:` / `except BaseException`。

## 权限治理

三道，全部复用 Hermes 自身的审批引擎，不重造：

1. **启动审批**（`require_launch_approval`，默认开）：顶层 launch 前——CLI 同步确认；
   gateway 发 approve/deny 按钮并阻塞；无人值守（无通道）则拒绝。嵌套 `workflow()`
   继承已审批的父 run，不再单独审批。
2. **子代理命令审批**（`child_approval_policy`）：子代理工具调用照常走 Hermes 审批
   引擎（危险命令检测、hardline 底线、permanent 白名单、yolo、gateway 异步审批）。
   本键只决定「被标记的命令本会提示人、但人不在场（后台 run）」时怎么办：`inherit`
   （跟随 Hermes `approvals.mode`，默认）/ `smart`（辅助 LLM 守卫，推荐无人值守）/
   `deny` / `approve` / `ask`（有通道问人，否则退化为 `ask_fallback`）。
3. **pre_tool_call 钩子**：后台子代理跑在脱离会话上下文的线程里，Hermes 自身审批会
   因缺上下文而误放行或挂起。钩子在 Hermes 的上下文分支前先施加上述策略；放行时还
   `approve_session()` 该模式，免得下游重新 gating 把命令变成无人可答的 pending。
   hardline 底线与 permanent 白名单始终生效。

## Transcript（从 state.db 重建执行链路）

子代理是独立 `AIAgent`，消息落在 Hermes 的 `SessionDB`（SQLite）。为让用户 / 主代理
看每个子代理的完整执行链路，运行时把这些消息导出成 `agent-<sessionId>.jsonl`
（+ 同名 `.meta.json` 边车）。

- **增量读**（`SessionTranscriptReader`）：直接读 `messages` / `sessions` 表。先用
  递归 CTE 解出**压缩血缘**（一个子代理被上下文压缩后派生新 session，串成一条
  lineage），再按 `(行数, min/max id, active 计数与 id 和)` 判断是否纯追加——是则只
  追加新消息，否则整体重建。
- **全量回退**：schema 不符 / 私有连接不可用 / 增量读异常时，退回公共 API
  `get_messages(include_inactive=True)`，按内容签名判断是否变化。
- **实时导出**（`LiveTranscriptExporter`）：每 0.5s 刷新活跃子代理，单 reader + 单
  writer 串行（避免多 SQLite 连接与原子临时文件竞争）；子代理转终态再 flush 一次；
  run 结束做一次带校验的最终重建。

`/workflows` 与面板据此展示每个子代理的 prompt、近期工具活动、产出，**不依赖**最终
output 文件。

## 沙箱与确定性

脚本经 AST 校验后以受限全局 `exec`，门的是**能力**而非**控制流**：`if/for/while/try`
允许（loop-until-budget / loop-until-dry 需要），但 import、文件/进程/网络、dunder
穿透、`eval/exec/compile/open/getattr…`、class 定义、动态调用目标一律拒绝；时间/随机
API 被禁（破坏 resume）。这是护栏，不是完美沙箱——真正隔离需子进程 + RPC（后续步骤）。

## Resume / 内容寻址缓存

`resumeFromRunId` 复用上一 run 中**未改动**的 `agent()` 结果。指纹按内容寻址
（prompt + 相关 opts），即便并发调度顺序变了，未变的调用仍命中。改脚本时尽量保留
靠前的稳定 `agent()` 调用：靠前改动顺流影响下游 prompt、降低复用，靠后改动保留更多
缓存。

## Token Budget

`budget.total` 解析自当前用户消息里的目标（`+500k`、`spend 2M tokens`、`use 1B
tokens`…），未写为 `None`。`spent()` 是本 run 已完成子代理的 token（输入+输出+推理）。
达到 `total` 后 `agent()` 抛 `WorkflowLimitExceeded`（run 级硬停）。作用域是**单 run**，
不是 Claude Code 的每轮共享池——独立工具该有的边界。工具入参 / `meta` / 配置 / 环境
都不能设 `total`。

## agentType / worktree / 命名工作流

- **agentType**：`agent(agentType="…")` 从工作流 agent 文件加载子代理指令。解析序：
  项目 `.hermes/dynamic-workflows/agents/<name>.{md,yaml,json}` → 用户
  `~/.hermes/dynamic-workflows/agents/<name>.…` → 插件内置 `agents/<name>.md`。
  Markdown 支持 YAML frontmatter（`model` / `toolsets` / `isolation`…）。内置：
  `explore`、`general-purpose`、`plan`、`verification`。
- **worktree**：`agent(isolation="worktree")` 在每子代理独立 git worktree 里跑，防
  并发改同一 checkout 冲突。是工作区隔离、非安全沙箱；用完默认删除（`keep_worktrees`
  关）。
- **命名工作流**：`/workflows <runId> save <name> [user|project]` 把脚本存为可复用
  命名工作流并注册 `/<name>` 斜杠命令（project 写 `.hermes/workflows/<name>.py`，
  user 写用户库）；`export` 则导出 markdown transcript。

## 控制（暂停/恢复/停止/重启）

独立面板 `hermes-workflows`（另开终端）通过**属主限定、带过期的请求/响应队列**（落在
插件 store 下，不开本地端口）把控制发回拥有该 run 的 Hermes 进程：

- `x` 停止该 run 并打断其活跃子代理。
- `p` 协作式暂停/恢复：暂停期间不起新子代理或后续 pipeline 阶段（在跑的可跑完），
  暂停时长不计入 deadline。
- `r` 用保存的脚本与 args 整体重启为新 Run ID 的全新 run。
- `s` 存一份 markdown transcript。

旧版本（无控制属主）的 run 用 `task_stop` / `/workflow-stop` 停。

## 配置

无独立配置文件：插件从 Hermes 的 `config.yaml` 读
`plugins.entries.dynamic-workflows.dynamic_workflows:` 一节，并支持
`HERMES_DYNAMIC_WORKFLOWS_*` 环境变量覆盖。键、默认值与含义见 README 的配置一节。
