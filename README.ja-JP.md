# Hermes Dynamic Workflows

> **[Hermes Agent](https://github.com/NousResearch/hermes-agent) 向けの Claude-Code スタイルの動的ワークフロー。**

[English](./README.md) | [简体中文](./README.zh-CN.md) | 日本語

Hermes で **動的ワークフロー（Dynamic Workflows）** を利用できるようになりました。モデルにサンドボックス化された Python
スクリプトをその場で書かせ、バックグラウンドランタイムで実行し、`agent()/parallel()/pipeline()` を使って
多数の独立したサブエージェントをオーケストレーションできます。コードベースの監査、大規模なマイグレーション、
クロスバリデーションを伴うリサーチに最適です。
[Dynamic Workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
にインスパイアされています。

https://github.com/user-attachments/assets/06ef3d0d-4d89-48c4-9851-e1cae690e9b0

## クイックスタート

1 行でインストールして有効化します。

```bash
hermes plugins install lingjiuu/hermes-dynamic-workflows --enable
```

> Gateway 利用者へ: インストール後に `hermes gateway restart` を実行してください。

インストールが完了したら、Hermes に「〜するワークフローを実行して」と伝えるだけで使えます。

### ライブダッシュボード（任意、別途セットアップが必要）

`hermes plugins install` はプラグインをクローンするだけで、コンソールスクリプトはインストールしません。
そのため、ダッシュボードコマンドは一度だけ別途インストールする必要があります。

```bash
python3 "${HERMES_HOME:-$HOME/.hermes}/plugins/dynamic-workflows/scripts/install-hermes-workflows.py"
# ~/.local/bin にインストールされます
```

その後、**別のターミナル**で `hermes-workflows` を実行すると、インタラクティブな
ダッシュボードが開きます。ここでは実行リスト、フェーズ／エージェントごとの進捗、各
サブエージェントのプロンプトと出力をリアルタイムで確認できます。

## 設定（任意）

プラグインは Hermes の `~/.hermes/config.yaml` から以下のセクションを読み込みます（すべてのキーは
`HERMES_DYNAMIC_WORKFLOWS_*` 環境変数でも上書きできます）。

```yaml
plugins:
  entries:
    dynamic-workflows:
      dynamic_workflows:
        concurrency: 8                # エージェントの最大同時実行数（デフォルト: min(16, cpu-2)）
        max_concurrency: 16           # 同時実行数のハードキャップ
        max_agents: 1000              # 1 回の実行あたりのエージェント総数の上限（暴走防止）
        workflow_timeout_seconds: 900 # 実行全体のウォールクロックタイムアウト（一時停止時間を除く）
        child_timeout_seconds: 300    # 単一の子エージェントのタイムアウト
        blocked_child_toolsets: [workflow, delegation, code_execution, memory, messaging, clarify]
                                      # 子エージェントの使用を禁止するツールセット
        default_child_toolsets: [web, file, terminal, skills]
                                      # 子エージェントのデフォルトツールセット（agentType が指定されていない場合に使用）
        keep_worktrees: false         # 各エージェントの git worktree を残すかどうか（デフォルトでは自動クリーンアップ）
        allow_model_override: true    # agent(model=...) によるモデルの上書きを許可するかどうか
        require_launch_approval: true # トップレベルのワークフロー起動前に確認を要求する（オンラインの人がいない場合は拒否）
        child_approval_policy: inherit # 子エージェントの承認ポリシー: inherit|smart|deny|approve|ask
        ask_fallback: smart           # "ask" で連絡先が誰もいない場合のフォールバック: smart|deny|approve
        notify_on_complete: true      # 完了時に起点となった CLI または gateway セッションへ通知する
        notify_result_preview_chars: 2000  # 通知での結果プレビューの切り詰め長（文字数）
```

## スクリプト API

ワークフロースクリプトは、最初のステートメントがリテラルの `meta` である非同期 Python のコードに
過ぎません。その後はサンドボックス化されたグローバルを使って子エージェントをオーケストレーションします。

```python
meta = {
    "name": "repo-audit",
    "description": "並列レビューの後に敵対的検証を行う",
    "phases": [{"title": "Review"}, {"title": "Verify"}],
}

# 各ターゲットは review → verify を独立して流れる
# (pipeline にはバリアがない: B がまだ review にいる間に A は verify にいられる)
findings = await pipeline(
    args["targets"],
    lambda t, _o, i: agent(f"バグをレビュー: {t}", {"label": f"review:{i}", "phase": "Review"}),
    lambda r, _o, i: agent(f"敵対的に検証: {json.dumps(r)}", {"label": f"verify:{i}", "phase": "Verify"}),
)
return await agent("検証済みの結果を統合する:\n" + json.dumps(findings))
```

- `agent(prompt, opts)` は子エージェントを起動します。`opts` には `schema`（構造化出力を強制）、
  `model`、`agentType`、`isolation="worktree"` を含めることができます。
- `pipeline`（デフォルト、バリアなし）／`parallel`（バリアあり）が並行処理を扱います。
  `phase`／`log` は進捗を報告し、`workflow()` は名前付きワークフローをインラインで実行し、`args` /
  `budget` は入力引数とトークン予算にアクセスします。

### エージェントタイプ

子エージェントのタイプはスクリプト内で `agentType` を使って指定します。省略した場合は
`general-purpose`（フルツールセット）がデフォルトになります。

| タイプ | ツールセット | 説明 |
|------|---------|-------------|
| `general-purpose` | `*`（すべての安全なツール） | デフォルト。コード検索、複雑な問題のリサーチ、複数ステップのタスクに適する |
| `explore` | 読み取り専用（read_file, search_files, terminal） | 高速なコードベース探索。ファイルの特定やキーワード検索に適する |
| `plan` | 読み取り専用（read_file, search_files, terminal） | ソフトウェアアーキテクチャ設計。ステップバイステップの実装計画を出力する |
| `verification` | web + file + terminal + browser | 実装の正しさを検証。build/test/lint を実行して PASS/FAIL を出力する |

エージェントタイプは優先順位順に 3 つの場所から解決されます（名前が衝突した場合は、
前方の場所が後方の場所を上書きします）。

1. `<project>/.hermes/dynamic-workflows/agents/*.md`  — プロジェクトレベル。現在のプロジェクトにのみ適用
2. `~/.hermes/dynamic-workflows/agents/*.md`          — ユーザーレベル。グローバルに適用
3. `<plugin>/hermes_dynamic_workflows/agents/*.md`    — 組み込みデフォルト（general-purpose/explore/plan/verification）

カスタムタイプを追加するには、上記 1 または 2 のディレクトリに次の形式で新しい `.md` ファイルを作成します。

```markdown
---
name: my-agent
description: "このエージェントが何のためのものかの短い説明。モデルがこれを使って適切なエージェントを自動選択します。"
model: inherit
toolsets: [web, file, terminal]
---

ここにエージェントのシステムプロンプトを記述し、その挙動、スタイル、制約を指示します。
```

`name` と `description` は必須です。`model` のデフォルトは `inherit`（現在のセッションの
モデルを継承）、`toolsets` のデフォルトはグローバルの `default_child_toolsets` です。
オプションフィールドとして `allowed_tools`、`disallowed_tools`、`isolation` もあります。

実行時、プラグインはスクリプトとすべての子エージェントの完全な実行トレース（トランスクリプト）を
永続化し、完了時に `<task-notification>` を会話に注入します。ポーリングは不要です。
履歴と詳細を表示するには `/workflows` を使用してください。

## ディープダイブ

実装の詳細（コア実行パス、ツールと完全な呼び出し結果、プロンプトキャッシュ、並行処理と制限、
権限ガバナンス、`state.db` からのトランスクリプトの再構築、サンドボックス化、レジューム…）については、
[TECHNICAL.md](./TECHNICAL.md) を参照してください。

## ライセンス

[MIT](./LICENSE)
