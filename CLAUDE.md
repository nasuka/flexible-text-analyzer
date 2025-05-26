# Mirai Analyzer

## プロジェクト概要
Streamlitを使用したテキスト分析Webアプリケーション。LLMを活用してテキストデータのトピック抽出・分類を行う高度な分析ツールです。

## 技術スタック
- **Framework**: Streamlit
- **Language**: Python 3.12+
- **AI/ML**: OpenAI API (GPT-4o), Structured Output
- **Data Analysis**: Pandas, Plotly
- **Text Processing**: YouTube Data API v3, MeCab
- **Dependency Management**: Rye
- **Linting/Formatting**: Ruff
- **Containerization**: Docker + Docker Compose

## 主要機能

### 1. データ取得 (`1_データ取得.py`)
- **YouTube コメント取得**: YouTube Data API v3を使用してコメントデータを収集
- **動画URLからのコメント抽出**: 指定した動画の全コメント・返信を取得
- **統計情報表示**: いいね数、投稿日時等の基本統計
- **CSV出力**: 取得したコメントデータをCSV形式でダウンロード

### 2. トピック抽出 (`2_トピック抽出.py`)
- **LLMトピック抽出**: OpenAI Structured Outputによる高精度トピック分析
- **自動トピック決定**: LLMがデータ内容に応じて最適なトピック数・サブトピック数を決定
- **全データ分析**: サンプリングではなく全テキストを対象とした包括的分析
- **感情分析**: テキストの感情傾向を同時に分析
- **可視化**: トピック分布、関係図、ワードクラウド等
- **セッション保持**: 分析結果がダウンロード後もリセットされない

### 3. トピック分類 (`3_トピック分類.py`)
- **自動テキスト分類**: 抽出されたトピックを使用してテキストを自動分類
- **信頼度評価**: 各分類の信頼度スコアを表示
- **分類理由表示**: なぜその分類になったかの詳細説明
- **柔軟なデータ入力**: 複数の方法でトピック定義とテキストデータを入力可能
- **結果出力**: 元のCSV + 分類結果の統合CSV出力

## プロジェクト構成
```
mirai-analyzer/
├── src/
│   ├── app.py                      # Streamlitアプリのメインファイル
│   └── pages/
│       ├── 1_データ取得.py         # YouTube コメント取得
│       ├── 2_トピック抽出.py       # LLMトピック抽出・感情分析
│       └── 3_トピック分類.py       # テキスト自動分類
├── pyproject.toml                  # プロジェクト設定・依存関係
├── requirements.lock               # 固定された依存関係
├── requirements-dev.lock           # 開発用依存関係
├── Dockerfile                      # Docker設定
├── docker-compose.yml              # Docker Compose設定
├── Makefile                       # 開発用コマンド
└── CLAUDE.md                      # プロジェクト情報（このファイル）
```

## コーディングガイドライン
* type hintはpython3.10以降に準拠
  * 利用禁止: List, Dict, Tuple, Optional
  * 利用可能: list, dict, tuple, type | None

## 開発コマンド

### アプリケーション起動
```bash
# 開発環境で起動（hot reload有効、推奨）
make up-dev

# 本番環境で起動
make up

# ローカルで起動
streamlit run src/app.py
```

### コード品質チェック
```bash
# Lint + Format
make check

# Lintのみ
make lint

# Formatのみ
make format
```

### Docker関連
```bash
make up-dev  # 開発環境コンテナ起動（hot reload有効）
make up      # 本番環境コンテナ起動
make down    # コンテナ停止
make build   # イメージビルド
make logs    # ログ確認
```

## アクセス情報
- **URL**: http://localhost:8501
- **健康チェック**: http://localhost:8501/_stcore/health

## 主要依存関係
```toml
[dependencies]
streamlit = ">=1.45.1"
pandas = ">=2.2.3"
plotly = ">=6.1.1"
openai = ">=1.0.0"
pydantic = ">=2.0.0"
google-api-python-client = ">=2.0.0"
python-dotenv = ">=1.1.0"
mecab-python3 = "*"
networkx = "*"
wordcloud = "*"
scikit-learn = "*"

[dev-dependencies]
ruff = ">=0.11.11"
```

## 環境変数設定
`.env` ファイルに以下の環境変数を設定してください：

```env
# OpenAI API Key (必須)
OPENAI_API_KEY=your_openai_api_key_here

# YouTube Data API Key (データ取得機能を使用する場合)
YOUTUBE_API_KEY=your_youtube_api_key_here
```

## 使用ワークフロー

### 基本的な分析フロー
1. **データ取得**: YouTube動画からコメントデータを収集
2. **トピック抽出**: LLMを使用してテキストからトピック・サブトピックを抽出
3. **トピック分類**: 抽出されたトピックを使用して新しいテキストデータを自動分類

### 代替ワークフロー
- **外部データ使用**: CSVファイルを直接アップロードしてトピック抽出・分類
- **JSONインポート**: 既存のトピック定義を使用して分類のみ実行

## 技術的特徴

### LLM統合
- **Structured Output**: Pydanticスキーマによる構造化された出力
- **高精度分析**: GPT-4oによる文脈理解に基づくトピック抽出
- **自動調整**: データ内容に応じたパラメータ自動決定

### データ可視化
- **Plotlyチャート**: インタラクティブなグラフ表示
- **ネットワーク図**: トピック間の関係性可視化
- **統計ダッシュボード**: 分析結果の包括的表示

### ユーザビリティ
- **セッション管理**: 分析結果の永続化
- **進捗表示**: 長時間処理の進捗バー
- **エラーハンドリング**: 詳細なエラーメッセージと回復手順

## 開発時の注意点
- **Pythonバージョン**: 3.12以上が必要
- **API制限**: OpenAI APIの使用量とコスト管理に注意
- **データサイズ**: 大量データ処理時のメモリ使用量に注意
- **環境変数**: APIキーは `.env` ファイルで管理
- **コード品質**: Ruffの設定に従ってフォーマット
- **Dockerコンテナ**: 本番環境ではDockerでの動作確認を推奨

## トラブルシューティング

### よくある問題
1. **OpenAI API エラー**: APIキーの確認、使用量制限の確認
2. **YouTube API エラー**: APIキーの確認、クォータ制限の確認
3. **メモリ不足**: データサイズの調整、バッチ処理の検討
4. **日本語処理エラー**: MeCabの正常インストール確認

### パフォーマンス最適化
- **データ制限**: 大量データは分割処理を推奨
- **API効率化**: バッチ処理による API 呼び出し最適化
- **キャッシュ活用**: セッション状態による結果保持