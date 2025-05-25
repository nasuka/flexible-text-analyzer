# Mirai Analyzer

## プロジェクト概要
Streamlitを使用したWebアプリケーション。現在は基本的な"Hello World"画面を表示。

## 技術スタック
- **Framework**: Streamlit
- **Language**: Python 3.12+
- **Dependency Management**: Rye
- **Linting/Formatting**: Ruff
- **Containerization**: Docker + Docker Compose

## プロジェクト構成
```
mirai-analyzer/
├── src/
│   ├── app.py              # Streamlitアプリのメインファイル
│   └── mirai_analyzer/     # パッケージディレクトリ
├── pyproject.toml          # プロジェクト設定・依存関係
├── requirements.lock       # 固定された依存関係
├── Dockerfile              # Docker設定
├── docker-compose.yml      # Docker Compose設定
└── Makefile               # 開発用コマンド
```

## 開発コマンド

### アプリケーション起動
```bash
# Dockerで起動 (推奨)
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
make up      # コンテナ起動
make down    # コンテナ停止
make build   # イメージビルド
make logs    # ログ確認
```

## アクセス情報
- **URL**: http://localhost:8501
- **健康チェック**: http://localhost:8501/_stcore/health

## 依存関係
- streamlit>=1.45.1
- pandas>=2.2.3
- plotly>=6.1.1
- python-dotenv>=1.1.0
- ruff>=0.11.11

## 開発時の注意点
- Pythonバージョン: 3.12以上が必要
- 環境変数は `.env` ファイルに記載
- コードはRuffの設定に従ってフォーマット
- Dockerコンテナでの動作確認を推奨