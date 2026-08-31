[English](architecture.md) | [日本語](architecture.ja.md)

# アーキテクチャ

## 世界状態 — `CultureWorld`

固定サイト型のモデルである．全セルが占有され，エージェントは移動せず，文化ベクトルのみが mutate する．したがって占有追跡型の `GridIndex` ではなく，`socsim-grid::CellGrid<Culture>` + 事前計算 `Adjacency` (axelrod1997 と同一) を採用する．

- `cells: CellGrid<Culture>` — `Culture = Vec<usize>`．セル値 = 文化ベクトル，フラット idx = `r*cols + c`．状態の単一真実源．
- `adjacency: Adjacency` — CSR のフォン・ノイマン (4 近傍) 表．`cells.grid()` から一度だけ構築．
- `n_features` (`F`), `n_traits` (`q`), `width`, `height`.
- LLM レイヤ: `personas: BTreeMap<AgentId, String>`, `memory: BTreeMap<AgentId, Vec<String>>` (古典変種では空)．
- `lc_history`, `gp_history` — 収束メカニズムが毎ラウンド push するバッファ．

`WorldState::agent_ids()` は `0..width*height` をソート済み `AgentId` で返す．

## メカニズム × フェーズ

| Mechanism | Phase | 役割 |
|-----------|-------|------|
| `ClassicalInteractionMechanism` | `Interaction` | 決定論的 Axelrod ベースライン．1 tick = `events_per_step` micro-event: サイト `s` + ランダム隣人 `nb` を選び `sim` を計算，確率 `sim` で差異特徴を 1 つ `nb` からコピー．LLM 不使用． |
| `LLMInteractionMechanism` | `Interaction` | YuLan-OneSim 変種．同じイベント駆動枠組みだが，採用判断 (採用するか/どの差異特徴か) を LLM に委ねる (ペルソナ + 自文化 + 隣人文化を与える)．LLM 呼び出しはここに閉じ，サイトの memory を更新． |
| `ConvergenceMechanism` | `PostStep` | 毎ステップ LC / GP を計算して world 履歴へ push，吸収状態 (全隣接 `sim ∈ {0,1}`) を検出し `request_stop`． |

`ClassicalInteractionMechanism` と `LLMInteractionMechanism` は **排他**で，ドライバが `config.provider` に応じてどちらか一方だけを追加する．これにより同一 world・同一指標で両者を直接比較できる．

## 更新セマンティクス

イベント駆動 (Axelrod / voter モデルの標準型) であり，YuLan-OneSim の非同期イベントバス型パラダイムと整合する．1 エンジン tick = `events_per_step` 回の micro-event をバッチ実行 (既定 = `n_sites`)．サイト選択は `ctx.rng` を使うため結果はスケジューラ非依存である．スケジューラは規約上 `RandomActivationScheduler`．

## RNG ストリーム (決定論)

```text
RNG_WORLD_INIT = 0   // 初期文化配置 + ペルソナ割当
RNG_ENGINE     = 1   // scheduler / engine / イベントのサイト・隣人抽選
```

`init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]))`，エンジン seed は `derive_seed(root, &[RNG_ENGINE])`．axelrod1997 / schelling1971 と同一規約．`run` サブコマンドは run ごとに `derive_seed(base, &[F, q, run])` を派生し，複数 run 平均を再現可能にする．

## 二層決定論

- **下層 (決定論的 socsim コア):** 文化初期化・サイト/隣人抽選・スケジュール・指標．シード固定で bit 再現．
- **上層 (非決定的 LLM):** `llm.rs` の `CachingClient<Box<dyn LlmClient>>` 経由で `LLMInteractionMechanism` に閉じる．本番は `FallbackClient<OllamaClient, OpenAiClient>` (Ollama 第一 → OpenAI フォールバック)，テストは `socsim_llm::mock::ScriptedClient` を注入．`temperature=0` + 固定 seed + プロンプト→応答キャッシュで再実行時に同一応答を再生．provider / model / temperature は run の `run.json` の `llm` ブロックに，呼び出し数と cache-hit 率は run スコープ指標 (`llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate`) に記録する．LLM を 1 度も叩かなかった run にはこれらの行も `llm` ブロックも無い — 分母 0 の率は «0» ではなく «定義できない» からである．

## 主要数式

文化ベクトル: `c_s = (c_s^1, …, c_s^F)`, `c_s^i ∈ {0,…,q-1}`.

相似度と古典相互作用確率:

```text
sim(s, nb) = |{ i : c_s^i = c_nb^i }| / F
P(interact | s, nb) = sim(s, nb)
```

吸収条件: 安定 ⇔ 全隣接 `(s, nb)` で `sim(s, nb) ∈ {0, 1}`.

付録 F の検証指標:

```text
LC = (1/|E|) Σ_{(i,j)∈E} |F_i ∩ F_j| / |F|       (局所収束; 隣接ペア平均類似度)
GP = |C| / N²                                     (大域分極; 文書記載どおり)
```

`E` は隣接エージェントペア集合，`|C|` は異文化領域数 (同一文化の連結クラスタ数)，`N` はエージェント総数．

## GP の不一致 (重要)

設計書は `GP = |C| / N²` と記載するが，既知の不一致を flag している: 付録 F は安定後 `GP ≈ 0.35–0.40` を報告するが，これは妥当な `N` では `|C| / N²` で **到達不能**である．`|C| ≤ N` より `|C| / N²` は `1/N` で上限される (例: `N = 100` → 最大 `0.01`)．したがって論文の `0.35–0.40` は **別の正規化** — 最も妥当には `|C| / N` (エージェントあたり領域数) — を使っている可能性が高い (小領域が多数残る場合この帯に入りうる)．

本実装は数式を **黙って「修正」しない**:

- `GP = |C| / N²` を文書どおりの `gp` フィールド/列として **そのまま実装**し，
- 補助列 `gp_per_agent = |C| / N` も **併記**する．

再現ヘルパと可視化は両者を `0.35–0.40` 帯と比較する．コード内注記は `metrics.rs::global_polarization` を参照．

## 中間スナップショット

`run --snapshot-interval N` で `N > 0` のとき，実行ドライバはラウンド 0・`N` ラウンドごと・最終ラウンドの文化グリッドを `SimulationResult::snapshots` に複製し，run の `artifacts/snapshots/culture_grid_round_<NNNNNN>.csv` (+ ラウンド一覧の `artifacts/snapshots/index.json`) に書き出す．Python の `animate` ツールがこれらを文化マップのモンタージュ + GIF に描画する．`N = 0` では最終グリッドのみ (既定)．

## 行動グラフ / ODD 概念エクスポート

YuLan-OneSim は自然言語シナリオから ODD プロトコル文書と内部の行動グラフ (agents → events → state updates) を構築し，エンジンが実行する．その *LLM 駆動構築パイプライン*の再現は本実装の対象外．明確に限定した **概念デモ**として，`run` はこの写像を逆向きに辿る: `odd::build_behavior_graph` が — **決定論的に，固定された `Config` + 配線済みメカニズムから，LLM なしで** — 7 つの ODD セクションとノード/エッジ行動グラフを含む構造的な `behavior_graph.json` を導出する．グラフは変種依存で，LLM 変種はペルソナ / メモリ / `llm_decision` ノードを追加し，ドライバが配線するメカニズムを正確に反映する．`behavior_graph.json` の `provenance` フィールドは，これが固定モデルの構造的記述であり LLM 合成物ではないことを正直に明記する．Python の `behavior-graph` ツールが図に描画する．

## 再現 & 比較ハーネス

- `reproduce` は付録 F / Table 7-2 の 4 条件 (F5q10, F5q15, F10q10, F15q15) を 10×10 グリッドで実行する (古典・オフライン・LLM 呼び出し 0)．親 run 1 本 + **条件ごとの子 run** (`subcommand=reproduce-condition`) の形で，子が試行を終端イベントに，条件の平均を run スコープ指標 (`mean_n_stable_regions` ほか) に，Axelrod の報告値を出典つきで `reference.csv` に持つ．許容幅は論文のものではなく**こちらが置いたもの**なので，PASS/off の判定とともに親の `artifacts/reproduce_verdicts.csv` に置く．Python の `reproduce` ツールが観測 vs 論文の図を描画する．
- `compare` は古典ベースラインと LLM 変種を一致条件 (同一グリッド / シード / ラウンド) で実行する．親 1 本 + **片側ごとの子 run** (`subcommand=compare-side`) の形にした．両側とも «同じ盤面を機構だけ変えて回した 1 本のシミュレーション» なので，それぞれが自分の `metrics.csv` と自分の `llm` ブロックを持つ．1 本の run に押し込むと指標の主キーが衝突し，1 つの run が 2 つのモデルを名乗ることになる．両側の差は «条件をまたいだ集約» なので親の `scope=sweep` 指標 (`delta_lc` ほか) に 1 度だけ置く．`--mock` は LLM 側を決定論的スクリプトクライアントにし比較全体をオフラインで実行する．実 LLM の数値はプロンプトキャッシュで擬似決定論化される．

## 出力

出力の置き場と同一性は [runvault](https://github.com/akitenkrad/rs-runvault) が持つ．1 回の実行が 1 つの run ディレクトリ `<results-root>/culture-llm/<run_slug>/` になり，命名は `Run::start` が行う．こちらでタイムスタンプ付きディレクトリも `latest` シンボリックリンクも作らない．run の場所は `runvault path --experiment culture-llm --latest --subcommand <sub>` で解決する．

- `config.json` — 封筒．実験条件は `parameters` の下．`llm_cache_path` は条件ではなく «置き場» なので `config_hash` から外してある．
- `run.json` — run の同一性: サブコマンド・シード・lineage・`llm` ブロック・再現実験のメタデータ．
- `metrics.csv` — long 形式 (`run_uid, step, step_unit, scope, name, value`)．ラウンドごと (`step_unit=round`, `scope=run`): `lc`, `gp`, `gp_per_agent`, `n_stable_regions`, `max_region_size`, `n_distinct_cultures`．step 無し: `converged`, `final_round`，LLM 経路では `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate`．
- `events.jsonl` — シミュレーション 1 本につき `terminal` 1 行 (`outcome` / `censored` / `budget` / `seed` + 最終指標) と，いつ観測したかを言う `observation` 行．
- `reference.csv` — 論文が印字した値と出典．持っているのは `reproduce` の子だけ．
- `artifacts/culture_grid_final.csv` — 最終文化グリッド (`row, col, culture`)，文化マップ可視化用．時間軸を持たない空間スナップショットで `culture` はラベルなので，指標ではなく表．
- `artifacts/behavior_graph.json` — 行動グラフ / ODD 概念エクスポート (毎回の `run`)．
- `artifacts/snapshots/culture_grid_round_<NNNNNN>.csv` + `artifacts/snapshots/index.json` — 中間文化グリッド (`--snapshot-interval N > 0` のときのみ)．
- `artifacts/reproduce_verdicts.csv` — 許容幅と PASS/off の判定 (`reproduce` の親のみ)．指標でも報告値でもない．
- `manifest.csv` / `status.json` — `finish()` が確定させる．後から描く図は run の *隣* (`<results-root>/culture-llm/figures/<run_slug>/`) に置き，manifest と食い違わないようにする．

サブコマンドと run の対応: `run` → run 1 本，`sweep` → 親 (`sweep`) + `(F, q)` セルごとの子 (`sweep-point`)，`reproduce` → 親 + 条件ごとの子 (`reproduce-condition`)，`compare` → 親 + 片側ごとの子 (`compare-side`)，`examples/mock_smoke` → run 1 本 (`mock-smoke`)．子が `run` を名乗ることは無いので，`runvault path --subcommand run` は一意である．

runvault 以前に書かれた `results/<timestamp>/` は書き換えていない．各ツールの `--results-dir` に直接渡せば従来どおり読める．

---
*This file was generated by Claude Code.*
