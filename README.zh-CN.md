# Hermes Dynamic Workflows

现在你可以在 Hermes 里使用 **Dynamic Workflows** 了：让模型现写一段受限 Python 脚本，
后台运行时执行它、用 `agent()/parallel()/pipeline()` 编排大量独立子代理——适合代码库
审计、大规模迁移、交叉验证的研究。参考自 [Dynamic Workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)。

## 快速开始

一行装好并启用：

```bash
hermes plugins install lingjiuu/hermes-dynamic-workflows --enable
```

> gateway 用户装完再 `hermes gateway restart`。

装完直接对 Hermes 说「用 workflow 跑一个 …」即可。

### 实时面板（可选，需单独一步）

`hermes plugins install` 只克隆插件、不安装它的 console 脚本，所以面板命令要单独装一次：

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/plugins/dynamic-workflows/scripts/install-hermes-workflows.py"
# 装到 ~/.local/bin
```

之后在**另一个终端**运行 `hermes-workflows`，打开交互式面板，可以实时查看 run 列表、各 phase/agent 进度、
每个子代理的 prompt 与产出。

## 配置（可选）

插件从 Hermes 的 `~/.hermes/config.yaml` 读下面这一节（每个键也支持
`HERMES_DYNAMIC_WORKFLOWS_*` 环境变量覆盖）：

```yaml
plugins:
  entries:
    dynamic-workflows:
      dynamic_workflows:
        concurrency: 8                # 最大并发 agent 数（默认 min(16, cpu-2)）
        max_concurrency: 16           # 并发上限硬限制
        max_agents: 1000              # 单个 run 的 agent 总数上限（防逃逸）
        workflow_timeout_seconds: 900 # 整个 run 的 wall-clock 超时（不含暂停时间）
        child_timeout_seconds: 300    # 单个子 agent 超时
        blocked_child_toolsets: [workflow, delegation, code_execution, memory, messaging, clarify]
                                      # 子 agent 禁止使用的 toolsets
        default_child_toolsets: [web, file, terminal, skills]
                                      # 子 agent 默认 toolset（不指定 agentType 时生效）
        keep_worktrees: false         # 是否保留 agent 的 git worktree（默认自动清理）
        allow_model_override: true    # 是否允许 agent(model=...) 指定模型
        require_launch_approval: true # 顶层 workflow 启动前需确认（无人在线则拒绝）
        child_approval_policy: inherit # 子 agent 审批策略: inherit|smart|deny|approve|ask
        ask_fallback: smart           # ask 无人可达时的降级: smart|deny|approve
        notify_on_complete: true      # 完成时通知发起 CLI 或 gateway 会话
        notify_result_preview_chars: 2000  # 通知中结果预览的截断字符数
```

## Script API

工作流脚本就是一段 async Python，首句是字面量 `meta`，之后用受限全局编排子代理：

```python
meta = {
    "name": "repo-audit",
    "description": "Parallel review, then adversarial verify",
    "phases": [{"title": "Review"}, {"title": "Verify"}],
}

# 每个目标独立流过 review → verify（pipeline 无栅栏：A 可在 verify 时 B 还在 review）
findings = await pipeline(
    args["targets"],
    lambda t, _o, i: agent(f"Review for bugs: {t}", {"label": f"review:{i}", "phase": "Review"}),
    lambda r, _o, i: agent(f"Verify adversarially: {json.dumps(r)}", {"label": f"verify:{i}", "phase": "Verify"}),
)
return await agent("Synthesize the verified findings:\n" + json.dumps(findings))
```

- `agent(prompt, opts)` 起一个子代理；`opts` 可带 `schema`（强制结构化输出）、`model`、
  `agentType`、`isolation="worktree"`。
- `pipeline`（默认，无栅栏）/ `parallel`（栅栏）做并发；`phase`/`log` 报告进度；
  `workflow()` 内联跑命名工作流；`args` / `budget` 取入参与 token 预算。

### Agent Type

脚本里通过 `agentType` 指定子代理类型，不填则默认 `general-purpose`（全工具集）:

| 类型 | 工具集 | 说明 |
|------|--------|------|
| `general-purpose` | `*`（全部安全工具） | 默认，适合搜索代码、研究复杂问题、多步任务 |
| `explore` | 只读（read_file, search_files, terminal） | 快速代码库探索，适合找文件、搜关键词 |
| `plan` | 只读（read_file, search_files, terminal） | 软件架构设计，输出分步实现方案 |
| `verification` | web + file + terminal + browser | 验证实现正确性，跑构建/测试/lint 出 PASS/FAIL |

Agent type 按优先级从三个位置查找（同名时前面的覆盖后面的）:

1. `<项目>/.hermes/dynamic-workflows/agents/*.md`   — 项目级，仅当前项目生效
2. `~/.hermes/dynamic-workflows/agents/*.md`        — 用户级，全局生效
3. `<插件>/hermes_dynamic_workflows/agents/*.md`     — 内置默认（general-purpose/explore/plan/verification）

加自定义类型:在 1 或 2 的目录下新建 `.md`，格式如下:

```markdown
---
name: my-agent
description: "简短描述这个 agent 的用途,模型会根据描述自动选择合适的 agent。"
model: inherit
toolsets: [web, file, terminal]
---

你可以在这里写 agent 的 system prompt,指导它的行为、风格和约束。
```
`name` 和 `description` 必填,`model` 默认 `inherit`(继承当前会话模型),
`toolsets` 默认走全局 `default_child_toolsets`,可选字段还有 `allowed_tools`、`disallowed_tools`、`isolation`。

运行时持久化脚本与每个子代理的完整执行链路（transcript），并在完成时把
`<task-notification>` 注入对话——无需轮询。用 `/workflows` 看历史与详情。

## 深入

实现细节（核心链路、工具与完整调用结果、prompt cache、并发与限额、权限治理、从
state.db 重建 transcript、沙箱、resume…）见 [TECHNICAL.md](./TECHNICAL.md)。

## License

[MIT](./LICENSE)
