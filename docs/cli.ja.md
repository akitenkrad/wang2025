[English](cli.md) | [日本語](cli.ja.md)

# CLI リファレンス

バイナリ: `culture-llm`．サブコマンド: `run`, `sweep`, `reproduce`, `compare`．

## `run`

単一設定で実行 (古典 / LLM 相互作用)．

| フラグ | 既定 | 意味 |
|------|---------|---------|
| `--provider` | `none` | `none` (古典・LLM 不使用) / `ollama` / `openai` |
| `--width` | `10` | グリッド幅 (列数) |
| `--height` | `10` | グリッド高さ (行数) |
| `--features`, `-f` | `5` | 特徴数 `F` |
| `--traits`, `-q` | `10` | 特性数 `q` |
| `--runs` | `1` | 独立試行数 (古典: 平均，LLM: 通常 1) |
| `--rounds` | `20000` | 最大エンジン tick 数 |
| `--events-per-step` | `0` | tick あたり micro-event 数 (`0` = n_sites) |
| `--snapshot-interval` | `0` | 中間文化グリッドのスナップショット間隔 (ラウンド; `0` = 最終のみ) |
| `--seed` | (ランダム) | 乱数シード (socsim コア層を支配) |
| `--temperature` | `0.0` | LLM 生成温度 |
| `--llm-seed` | `0` | LLM 生成シード (バックエンド) |
| `--cache-path` | `.llm_cache/cache.json` | プロンプト→応答キャッシュ (LLM 経路) |
| `--output-dir` | `results` | results ルート (runvault が `<root>/culture-llm/<run_slug>/` に書く) |

例:

```bash
# 古典 Table 7-2 ベースライン (LLM 不使用・LLM 呼び出し 0)
cargo run --release -- run --provider none --width 10 --height 10 --features 5 --traits 10 --runs 30 --seed 42

# LLM 変種・小グリッド・Ollama 第一候補
OLLAMA_MODEL=llama3.2:latest cargo run --release -- run --provider ollama --width 5 --height 2 --features 5 --traits 5 --rounds 100 --seed 42

# 中間文化グリッドのスナップショット (アニメーション用)
cargo run --release -- run --provider none --width 10 --height 10 --features 5 --traits 8 --snapshot-interval 50 --seed 42
```

`run` は毎回 `artifacts/behavior_graph.json` (モデルから導出した行動グラフ / ODD 概念エクスポート) も書き出す．`--snapshot-interval N > 0` のとき，中間文化グリッドを `artifacts/snapshots/culture_grid_round_<NNNNNN>.csv` (`artifacts/snapshots/index.json` で索引付け) に書き出す．ラウンド 0 と最終ラウンドは常に含む．

`--runs N` は N 本回して **最後の 1 本**の詳細を記録する (移行前と同じ)．どの試行が残るかを決めるので `runs` は条件の一部であり `parameters` に入れてある．`master_seed` はその試行を実際に支配したシード，`replicate_index` は `N-1` で，コマンドラインで与えた根のシードは `/parameters.seed` に残る．`--seed` を省いてもシードが失われることは無くなった — 1 つ引いて記録し，それを使う．

## `sweep`

特徴数 `F` × 特性数 `q` を走査する．グリッド定義を持つ掃引親 run (`subcommand=sweep`) 1 本と，**`(F, q)` セルごとの子 run** (`subcommand=sweep-point`) からなる．各セルの `runs` 本の試行は子の `events.jsonl` の `terminal` 行 1 本ずつになり，セルの集約 (`n_units`, `mean_n_stable_regions`, `mean_lc` ほか) は子の run スコープ指標になる．`sweep_summary.csv` は書かない — 1 行 1 試行の表は `culture-llm-tools visualize-sweep` が子から組み直す．

| フラグ | 既定 | 意味 |
|------|---------|---------|
| `--provider` | `none` | 古典 / LLM |
| `--width` / `--height` | `10` / `10` | グリッドサイズ |
| `--features-min/max/step` | `5` / `15` / `5` | 特徴数の範囲 (両端含む) |
| `--traits-min/max/step` | `5` / `15` / `5` | 特性数の範囲 (両端含む) |
| `--runs` | `30` | `(F, q)` あたり試行数 |
| `--rounds` | `20000` | 最大エンジン tick 数 |
| `--events-per-step` | `0` | tick あたり micro-event 数 (`0` = n_sites) |
| `--snapshot-interval` | `10` | sweep_config に記録 |
| `--seed` | `42` | シード基点 (各 run で独立に派生) |
| `--temperature` / `--llm-seed` / `--cache-path` | `run` と同じ | LLM 設定 |
| `--output-dir` | `results` | results ルート (runvault が `<root>/culture-llm/<run_slug>/` に書く) |

```bash
cargo run --release -- sweep --provider none \
    --features-min 5 --features-max 15 --features-step 5 \
    --traits-min   5 --traits-max   15 --traits-step   5 \
    --runs 30 --seed 42
```

## `reproduce`

付録 F / Axelrod Table 7-2 一括再現．4 条件 (F5q10, F5q15, F10q10, F15q15) を 10×10 グリッドで各 `--runs` 回 (古典 `--provider none`・オフライン・LLM 呼び出し 0) 実行し，親 run 1 本 + **条件ごとの子 run** (`subcommand=reproduce-condition`) の形で記録する．子は試行を `terminal` 行に，条件の平均を run スコープ指標 (`mean_n_stable_regions`, `mean_lc` ほか) に，Axelrod の報告値を出典つきで `reference.csv` に持つ．許容幅と PASS/off の判定は論文のものではなくこちらが置いたものなので，親の `artifacts/reproduce_verdicts.csv` とコンソールに残す．Python の `culture-llm-tools reproduce` が観測 vs 論文の図を親の隣に描画する．

| フラグ | 既定 | 意味 |
|------|---------|---------|
| `--provider` | `none` | 古典ベースライン (オフライン検証可) |
| `--runs` | `30` | 条件あたり試行数 (平均) |
| `--rounds` | `20000` | 試行あたり最大エンジン tick 数 |
| `--seed` | `42` | シード基底 (試行ごとに独立シード派生) |
| `--quick` | off | 高速スモーク (`runs=5`, `rounds ≤ 5000`)・検証用ではない |
| `--output-dir` | `results` | results ルート |

```bash
cargo run --release -- reproduce --runs 30 --seed 42
cargo run --release -- reproduce --quick          # 高速エンドツーエンドスモーク
```

## `compare`

古典 (`--provider none`) 対 LLM の定量比較を **一致条件** (同一グリッド / シード / ラウンド) で実行．親 run (`subcommand=compare`) 1 本 + **片側ごとの子 run** (`subcommand=compare-side`) の形にする．両側とも «同じ盤面を機構だけ変えて回した 1 本のシミュレーション» なので，それぞれが自分のラウンドごとの `metrics.csv`・自分の `terminal` 行・(LLM 側は) 自分の `llm` ブロックを持つ．差 (`delta_n_stable_regions`, `delta_lc`, `delta_gp`, `delta_gp_per_agent`, `delta_final_round`) は親の `scope=sweep` 指標に 1 度だけ置く．`--mock` で LLM 側を決定論的スクリプトクライアント (ネットワーク不要) にすると比較全体がオフラインで完結する．`--mock` なしでは LLM 側は実 env 構築クライアント．

| フラグ | 既定 | 意味 |
|------|---------|---------|
| `--llm-provider` | `ollama` | 古典ベースラインと比較する LLM プロバイダ |
| `--mock` | off | 決定論的スクリプト LLM クライアント (オフライン; CI / サンドボックス) |
| `--width` / `--height` | `5` / `4` | 一致グリッドサイズ |
| `--features` / `--traits` | `5` / `5` | 一致 `F` / `q` |
| `--rounds` | `100` | 最大エンジン tick 数 |
| `--seed` | `42` | 共有シード (両側) |
| `--temperature` / `--llm-seed` / `--cache-path` | `run` と同じ | LLM 設定 (実経路) |
| `--output-dir` | `results` | results ルート |

```bash
# オフライン (スクリプト mock LLM): 古典は実行・LLM は構造的
cargo run --release -- compare --mock --features 5 --traits 5 --rounds 100 --seed 42

# 実 LLM (到達可能な Ollama / OpenAI バックエンドが必要)
OLLAMA_MODEL=llama3.2:latest cargo run --release -- compare --llm-provider ollama --rounds 100 --seed 42
```

---
*This file was generated by Claude Code.*
