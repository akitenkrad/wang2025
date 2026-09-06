<p align="center"><img src="docs/assets/hero.svg" width="100%"></p>

[English](README.md) | **日本語**

# YuLan-OneSim — Axelrod 文化伝播 (Wang et al. 2025)

YuLan-OneSim (Wang, Gao, Bo, Chen & Wen, 2025; arXiv:2505.07581; NeurIPS 2025 Workshop SEA) の **一つの具体シナリオ** の再現実装である．対象は，論文付録 F で定量検証されている **Axelrod (1997) 文化伝播** 実験である．YuLan-OneSim 全体は次世代 LLM ソーシャルシミュレータ (コード不要のシナリオ構築・50 種の組込シナリオ・自己進化・最大 10 万エージェント・AI ソーシャルリサーチャ) であり，そのすべての再現は対象外とする．本リポジトリは姉妹実装 [`axelrod1997`](../axelrod1997) を拡張し，**同一の socsim `WorldState` 上** で **古典的・決定論的ベースライン** と **LLM 駆動の文化採用変種** を比較する．

各グリッドサイトは `F` 個の特徴からなる文化ベクトルを持ち，各特徴は `0..q` の特性値を取る．モデルはイベント駆動で，1 エンジン tick = `events_per_step` 回の micro-event (既定 = サイト数)．各イベントでサイト `s` とランダムなフォン・ノイマン隣人 `nb` を抽出し，相似度 `sim = 一致特徴数 / F` を計算し，確率 `sim` で差異特徴を 1 つ `nb` から採用する．全隣接ペアで `sim ∈ {0, 1}` になると吸収状態である．Axelrod の核心的知見 — 局所相互作用が *局所収束* を促す一方で *大域分極* を保つ — がこの規則から創発する．

`--provider` で **排他的** な 2 つの相互作用メカニズムを選択する:

- `--provider none` → `ClassicalInteractionMechanism`: 決定論的な Axelrod ベースライン (LLM 不使用)．**既定**で，Axelrod Table 7-2 の数値を再現する経路．
- `--provider ollama|openai` → `LLMInteractionMechanism`: YuLan-OneSim 変種．サイトのペルソナ・自文化ベクトル・隣人の文化を与え，採用するか/どの特徴を採用するかを LLM が決める．

## 二層決定論 (最初に読むこと)

LLM 出力は socsim の bit 再現性の **外側** にあるため，設計を 2 層に分ける:

- **決定論的 socsim コア** — 文化初期化・サイト/隣人抽選 (`ctx.rng`, ChaCha20)・スケジュール・指標・収束判定．シードを固定すれば bit 単位で再現する．古典プロバイダは完全にこの層に閉じ，LLM 呼び出しは **0 回**．
- **非決定的 LLM レイヤ** — 採用判断．`socsim-llm` の `CachingClient` (`hash(prompt+model)` → 応答キャッシュ)・`temperature=0`・固定 seed で擬似決定論化する．プロバイダ順序は `socsim-llm` の `FallbackClient` による **Ollama 第一 → OpenAI フォールバック**．

再現性の本体はモデルではなく **キャッシュ** である．LLM を使った実行は provider / model / temperature を `run.json` の `llm` ブロックに，呼び出し数と cache-hit 率を run スコープ指標に記録する．LLM を叩かなかった run はどちらも持たない．ローカル既定モデル (`llama3.2`) は論文と異なるため，LLM 再現目標は **定性的** (モノカルチャ⇄ポリカルチャ転移が同じ) とし，**古典**経路は **定量的** に再現する．

## インストールとクイックスタート

```bash
# Rust シミュレーションをビルド (socsim と socsim-llm の Ollama+OpenAI バックエンドを取得)
cargo build --release

# === 古典 (LLM 不使用) ベースライン — Axelrod Table 7-2 を再現 ===
cargo run --release -- run --provider none --width 10 --height 10 --features 5 --traits 10 --runs 30 --seed 42

# === LLM 変種 (Ollama 第一候補) — 小グリッド ===
#   ollama pull llama3.2:latest
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=llama3.2:latest
cargo run --release -- run --provider ollama --width 5 --height 2 --features 5 --traits 5 --rounds 100 --seed 42

# === 感度分析スイープ (古典 F×q) ===
cargo run --release -- sweep --provider none \
    --features-min 5 --features-max 15 --features-step 5 \
    --traits-min   5 --traits-max   15 --traits-step   5 \
    --runs 30 --seed 42

# === 付録 F / Table 7-2 一括再現 (古典・オフライン) ===
cargo run --release -- reproduce --runs 30 --seed 42        # 観測 vs 論文 LC/GP + PASS/off

# === 古典 vs LLM 定量比較 (オフライン mock LLM) ===
cargo run --release -- compare --mock --features 5 --traits 5 --rounds 100 --seed 42

# Python 可視化ツール (ワークスペースルートで)
uv sync
uv run culture-llm-tools visualize                 # 文化マップ + LC/GP 時系列
uv run culture-llm-tools visualize-sweep           # F×q ヒートマップ
uv run culture-llm-tools show-experiment-settings  # 実験条件 + LLM の来歴
uv run culture-llm-tools reproduce                 # Table 7-2 観測 vs 論文の図
uv run culture-llm-tools animate                   # 中間文化マップアニメ / モンタージュ
uv run culture-llm-tools behavior-graph            # 行動グラフ / ODD 概念図
uv run culture-llm-tools compare-report            # 古典 vs LLM 比較図
```

LLM **パイプライン** のオフライン (LLM 不要) スモークは scripted mock クライアント経由で実行できる:

```bash
cargo run --release --example mock_smoke -- results
```

## 結果の置き場

記録は [runvault](https://github.com/akitenkrad/rs-runvault) に預けている．1 回の実行が 1 つの run ディレクトリ `results/culture-llm/<run_slug>/` になり，`config.json` (実験条件)・`run.json` (同一性・シード・lineage・`llm` ブロック)・`metrics.csv` (ラウンドごとと run 全体の数値，long 形式)・`events.jsonl` (シミュレーション 1 本につき `terminal` 1 行)・`artifacts/` (文化グリッド・スナップショット・ODD エクスポート) を持つ．掃引・再現バッチ・比較はいずれも «親 run 1 本 + セル / 条件 / 片側ごとの子 run» の形になる．run の場所は次で分かる:

```bash
runvault path --experiment culture-llm --latest --subcommand run --standalone
```

Python ツールも `--results-dir` を省略すると同じ問いを runvault に投げる．詳細は [アーキテクチャ § 出力](docs/architecture.ja.md#出力)．

## ドキュメント

- [アーキテクチャ](docs/architecture.ja.md) — 世界状態・メカニズム・二層決定論・スナップショット・行動グラフ / ODD 概念エクスポート・GP の不一致注記
- [CLI リファレンス](docs/cli.ja.md) — `run` / `sweep` / `reproduce` / `compare` のフラグ
- [再現](docs/reproduction.ja.md) — Axelrod Table 7-2 の数値・付録 F LC/GP・`reproduce` / `compare` ハーネス
- [可視化](docs/visualization.ja.md) — Python ツールと出力 (`animate`・`behavior-graph`・`compare-report` を含む)

## 参考文献

- Wang, L., Gao, H., Bo, X., Chen, X., & Wen, J.-R. (2025). *YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language Models.* arXiv:2505.07581.
- Axelrod, R. (1997). *The Dissemination of Culture: A Model with Local Convergence and Global Polarization.* Journal of Conflict Resolution, 41(2), 203–226.
- シミュレーション基盤: [socsim (rs-social-simulation-tools)](https://github.com/akitenkrad/rs-social-simulation-tools).

## ライセンス

MIT — [LICENSE](LICENSE) を参照．
