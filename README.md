# Text Analyzer

## 📋 プロジェクト概要

Streamlitを使用したテキスト分析Webアプリケーション。LLMを活用してテキストデータのトピック抽出・分類を行う高度な分析ツールです。

### 🌟 主な特徴

- **マルチプロバイダー対応**: OpenAI、OpenRouter（Gemini）
- **Structured Output**: 構造化された高精度な分析結果
- **リアルタイム分析**: 動的なトピック定義と即座の分類
- **可視化機能**: インタラクティブなグラフとネットワーク図
- **並列処理**: 大量データの高速処理


## 🚀 クイックスタート

### 前提条件

- Python 3.12+
- Rye（推奨）またはPip

### インストール

```bash
# リポジトリをクローン
git clone <repository-url>
cd text-analyzer

# Ryeを使用（推奨）
rye sync

# または、pipを使用
pip install -r requirements.lock
```

### 環境変数設定

`.env` ファイルを作成し、以下の設定を追加：

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# OpenRouter API Key（Geminiモデル使用時）
OPENROUTER_API_KEY=your_openrouter_api_key_here

# YouTube Data API Key（データ取得機能使用時）
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### アプリケーション起動

```bash
# 開発環境で起動（hot reload有効、推奨）
make up-dev

# またはローカルで起動
streamlit run src/app.py
```

アプリケーションは http://localhost:8501 でアクセス可能です。

## 🏗️ アーキテクチャ

### プロジェクト構成

```
text-analyzer/
├── src/
│   ├── app.py                      # Streamlitメインアプリ
│   ├── pages/                      # ページモジュール
│   │   ├── 1_データ取得.py         # YouTube コメント取得
│   │   ├── 2_トピック抽出.py       # LLMトピック抽出・感情分析
│   │   └── 3_トピック分類.py       # テキスト自動分類
│   ├── schema/                     # Pydanticスキーマ
│   │   ├── topic.py               # トピック分析関連
│   │   ├── classification.py      # 分類関連
│   │   └── llm_providers.py       # LLMプロバイダー定義
│   └── services/                   # ビジネスロジック
│       ├── llm.py                 # 共通LLMクライアント
│       ├── topic_extractor.py     # トピック抽出サービス
│       ├── topic_classifier.py    # トピック分類サービス
│       ├── youtube_fetcher.py     # YouTube API連携
│       └── text_column_estimator.py # テキストカラム推定
├── pyproject.toml                  # プロジェクト設定
├── requirements.lock               # 固定された依存関係
├── Dockerfile                      # Docker設定
├── docker-compose.yml              # Docker Compose設定
└── Makefile                       # 開発用コマンド
```

### 技術スタック

- **Framework**: Streamlit
- **Language**: Python 3.12+
- **AI/ML**: 
  - OpenAI API (GPT-4o, GPT-4o Mini)
  - OpenRouter API (Gemini 2.5 Flash, Gemini 2.5 Pro)
  - Structured Output / JSON Schema
- **Data Analysis**: Pandas, Plotly, NetworkX
- **Text Processing**: YouTube Data API v3, MeCab
- **Development**: Rye, Ruff, Docker

## 🎯 主要機能

### 1. データ取得

- **YouTube コメント取得**: 動画URLから全コメント・返信を収集
- **統計情報表示**: いいね数、投稿日時等の基本統計
- **CSV出力**: 構造化されたコメントデータの出力

### 2. トピック抽出

- **LLMトピック抽出**: Structured Outputによる高精度分析
- **自動最適化**: データに応じた最適なトピック数の自動決定
- **階層構造**: メイントピック + サブトピックの2層構造
- **感情分析**: ポジティブ・ネガティブ・中立の感情傾向分析
- **可視化**: 
  - トピック分布グラフ
  - ネットワーク関係図
  - 感情分析円グラフ

### 3. トピック分類

#### 多様な入力方法
- **トピック分析結果**: 抽出結果を直接利用
- **ページ上定義**: 動的フォームでトピック定義
- **JSONファイル**: 既存定義のインポート
- **組み合わせ**: CSVとJSONの柔軟な組み合わせ

#### 高度な分析機能
- **データ説明対応**: コンテキスト情報による精度向上
- **サブトピック分類**: オプションでサブトピック分類
- **信頼度評価**: 各分類の信頼度スコア
- **分類理由**: なぜその分類になったかの説明
- **並列処理**: バッチ処理による高速化
- **進捗表示**: リアルタイム進捗確認

#### 可視化・出力
- **分類統計**: トピック分布、信頼度分布
- **関係マトリックス**: トピック-サブトピック関係
- **CSV出力**: 元データ + 分類結果の統合
- **JSON出力**: 構造化された分析結果

### 4. 共通機能

- **マルチプロバイダー**: OpenAI、OpenRouter対応
- **テキストカラム推定**: 日本語テキストの自動検出
- **セッション管理**: 分析結果の永続化
- **エラーハンドリング**: 詳細なエラー情報と回復手順

## 🔧 開発者向け

### 開発コマンド

```bash
# アプリケーション起動
make up-dev              # 開発環境（hot reload有効）
make up                  # 本番環境
streamlit run src/app.py # ローカル起動

# コード品質
make check               # Lint + Format
make lint                # Lintのみ
make format              # Formatのみ

# Docker
make build               # イメージビルド
make down                # コンテナ停止
make logs                # ログ確認
```

### コーディングガイドライン

- **Type hints**: Python 3.12以降の記法を使用
- **利用禁止**: `List`, `Dict`, `Tuple`, `Optional`
- **利用推奨**: `list`, `dict`, `tuple`, `str | None`
- **Structured Output**: Pydanticモデルによる型安全性
- **エラーハンドリング**: 適切な例外処理とユーザーフィードバック

### API制限と最適化

- **バッチ処理**: 大量データの効率的処理
- **並列実行**: ThreadPoolExecutorによる高速化
- **トークン管理**: 推定トークン数表示
- **進捗表示**: 長時間処理の進捗確認

## 📊 使用例

### 基本的なワークフロー

1. **データ収集**: YouTube動画からコメントを取得
2. **トピック抽出**: LLMによる自動トピック分析
3. **トピック分類**: 新しいテキストデータの自動分類

### 応用例

- **顧客フィードバック分析**: 商品レビューの構造化分析
- **ソーシャルメディア分析**: SNS投稿の感情・トピック分析
- **カスタマーサポート**: 問い合わせの自動分類
- **コンテンツ分析**: ブログ・記事のトピック整理

## 🔐 セキュリティ

- **APIキー管理**: 環境変数による安全な管理
- **データプライバシー**: ローカル処理、外部送信最小化
- **入力検証**: 適切な入力値検証とサニタイゼーション

## 🐛 トラブルシューティング

### よくある問題

#### APIエラー
- **OpenAI**: APIキー確認、使用量制限確認
- **OpenRouter**: APIキー形式確認（`sk-or-`で開始）
- **YouTube**: APIキー確認、クォータ制限確認

#### パフォーマンス
- **メモリ不足**: データサイズ調整、バッチサイズ縮小
- **API制限**: 並列処理数調整、リクエスト間隔調整
- **処理時間**: バッチサイズとワーカー数の最適化

#### 日本語処理
- **MeCab**: 正常インストール確認
- **文字エンコーディング**: UTF-8設定確認

## 📈 パフォーマンス最適化

- **データ制限**: 大量データは分割処理を推奨
- **API効率化**: バッチ処理によるAPI呼び出し最適化
- **キャッシュ活用**: セッション状態による結果保持
- **並列処理**: ThreadPoolExecutorによる高速化

## 🤝 コントリビューション

1. フォークしてブランチ作成
2. 変更を実装
3. テスト実行: `make check`
4. プルリクエスト作成

## 📄 ライセンス

MIT License

## 🙋‍♀️ サポート

- Issues: GitHubのIssuesページ
- ドキュメント: `CLAUDE.md`の詳細情報
- 健康チェック: http://localhost:8501/_stcore/health