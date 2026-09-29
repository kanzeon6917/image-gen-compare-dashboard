# Image Generation Compare Dashboard

Stable Diffusionをローカル環境で動作させ、生成条件による画像の違いを比較・検証するためのプロジェクトです。

現在は、NVIDIA GPUを使用したローカル画像生成環境まで構築しています。

## 概要

画像生成では、同じプロンプトを使用していても、以下のような条件によって生成結果が変化します。

- プロンプト
- ネガティブプロンプト
- Seed
- Guidance Scale
- 推論ステップ数
- 画像サイズ

本プロジェクトでは、これらの条件を変更しながら生成結果を比較し、画像生成モデルの制御方法や各パラメータの影響を確認できるツールを目指します。

## 現在の実装状況

現時点では、以下まで動作確認済みです。

- Stable Diffusionによる画像生成
- NVIDIA GPUを使用したローカル推論
- PyTorchからのCUDA利用
- Hugging Face Diffusersを使用した推論
- uvによるPythonおよび依存パッケージ管理

### 動作確認環境

- Windows 11
- WSL2
- Ubuntu 24.04
- Python 3.12
- NVIDIA GeForce RTX 3060 12GB
- PyTorch + CUDA

## 使用技術

- Python
- PyTorch
- Hugging Face Diffusers
- Transformers
- Accelerate
- Safetensors
- uv

今後、以下も利用する予定です。

- Gradio
- ControlNet

## セットアップ

### 1. リポジトリを取得

```bash
git clone <YOUR_REPOSITORY_URL>
cd image-gen-compare-dashboard
```

### 2. uvをインストール

uvがインストールされていない場合は、以下を実行します。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

必要に応じてシェルを再起動するか、以下を実行します。

```bash
source ~/.bashrc
```

インストール確認：

```bash
uv --version
```

### 3. 依存パッケージをインストール

```bash
uv sync
```

### 4. GPUの認識を確認

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

実行例：

```text
PyTorch: 2.x.x+cuXXX
CUDA: XX.X
Available: True
GPU: NVIDIA GeForce RTX 3060
```

`Available: True` になっていれば、PyTorchからCUDAを利用できます。

## 画像生成

比較用 UI は次のコマンドで起動します。

```bash
uv run python app.py
```

`Compare parameter` で Guidance Scale または Steps を選び、比較候補を
カンマ区切りで入力します。比較しないパラメーターは固定スライダーで指定します。
各候補は同じ Prompt・Seed で生成されます。

- Guidance Scale: 0〜20
- Steps: 1〜100 の整数
- 比較候補: 最大20個
- Seed: 0〜4294967295 の整数

モデルは最初の Generate 時に読み込みます。画像生成には CUDA 対応 GPU が必要です。
結果は実行ごとに `outputs/<日時>_<一意なID>/` に保存されます。
画像ごとに同名の JSON ファイルを保存し、モデル ID・プロンプト・Seed・生成パラメーターを記録します。
途中で生成が失敗した場合も、保存済みの結果は残ります。

単体の生成スクリプトを使う場合は、
以下を実行します。

```bash
uv run python test_generate.py
```

生成された画像は `outputs/` 以下に保存されます。

## プロジェクト構成

現時点では、以下のような構成です。

```text
image-gen-compare-dashboard/
├── app.py          # Gradio UI と入力・結果の受け渡し
├── comparison.py   # 入力検証と比較条件の作成（GPU 不要）
├── generation.py   # モデル読み込み・生成・結果保存
├── test_generate.py
├── pyproject.toml
├── uv.lock
├── README.md
└── outputs/
```

`outputs/` は生成結果を保存するディレクトリで、Gitの管理対象から除外します。
