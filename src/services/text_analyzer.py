"""テキスト分析サービス

LLMを使用したテキスト分析機能を提供する
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from schema.llm_providers import LLMModel, LLMProvider
from services.llm import LLMClient


@dataclass
class AnalysisSettings:
    """分析設定"""

    provider: LLMProvider
    model: LLMModel
    api_key: str
    temperature: float = 0.3
    max_tokens: int = 3000


@dataclass
class AnalysisResult:
    """分析結果"""

    query: str
    result: str
    data: pd.DataFrame
    text_column: str
    metadata_columns: list[str]
    settings: AnalysisSettings
    stats: dict[str, Any]


class TextAnalyzer:
    """テキスト分析サービス"""

    def __init__(self, settings: AnalysisSettings):
        """初期化

        Args:
            settings: 分析設定
        """
        self.settings = settings
        self.llm_client = LLMClient(
            api_key=settings.api_key, model=settings.model, provider=settings.provider
        )

    def analyze_text(
        self,
        data: pd.DataFrame,
        text_column: str,
        metadata_columns: list[str],
        query: str,
        data_limit: int | None = None,
    ) -> AnalysisResult | None:
        """テキストデータを分析する

        Args:
            data: 分析対象データ
            text_column: テキストカラム名
            metadata_columns: メタデータカラム名のリスト
            query: 分析クエリ
            data_limit: 分析対象データの制限数

        Returns:
            分析結果、失敗時はNone
        """
        # データの準備
        analysis_df = data.head(data_limit) if data_limit else data

        # 基本統計の計算
        stats = self._calculate_stats(analysis_df, text_column)

        # 分析プロンプトの構築
        prompt = self._build_analysis_prompt(
            analysis_df, text_column, metadata_columns, query
        )

        # LLM分析の実行
        analysis_result = self.llm_client.simple_completion(
            prompt=prompt,
            system_message="あなたはデータ分析の専門家です。提供されたテキストデータを詳細に分析し、有用な洞察を提供してください。",
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
        )

        if not analysis_result:
            return None

        return AnalysisResult(
            query=query,
            result=analysis_result,
            data=analysis_df,
            text_column=text_column,
            metadata_columns=metadata_columns,
            settings=self.settings,
            stats=stats,
        )

    def estimate_tokens(
        self, data: pd.DataFrame, text_column: str, metadata_columns: list[str]
    ) -> int:
        """トークン数を予測する

        Args:
            data: 分析対象データ
            text_column: テキストカラム名
            metadata_columns: メタデータカラム名のリスト

        Returns:
            予測トークン数
        """
        total_chars = sum(len(str(row[text_column])) for _, row in data.iterrows())
        for col in metadata_columns:
            if col in data.columns:
                total_chars += sum(len(str(row[col])) for _, row in data.iterrows())

        # 文字数をトークン数に変換（日本語の場合、おおよそ3文字で1トークン）
        return total_chars // 3

    def _calculate_stats(self, data: pd.DataFrame, text_column: str) -> dict[str, Any]:
        """基本統計を計算する

        Args:
            data: 分析対象データ
            text_column: テキストカラム名

        Returns:
            統計情報
        """
        text_lengths = data[text_column].str.len()
        return {
            "total_count": int(len(data)),
            "avg_length": float(text_lengths.mean()),
            "unique_count": int(data[text_column].nunique()),
            "min_length": int(text_lengths.min()),
            "max_length": int(text_lengths.max()),
        }

    def _build_analysis_prompt(
        self,
        data: pd.DataFrame,
        text_column: str,
        metadata_columns: list[str],
        query: str,
    ) -> str:
        """分析用プロンプトを構築する

        Args:
            data: 分析対象データ
            text_column: テキストカラム名
            metadata_columns: メタデータカラム名のリスト
            query: 分析クエリ

        Returns:
            構築されたプロンプト
        """
        prompt = f"""以下のテキストデータを分析してください。

分析指示: {query}

データ一覧:
"""

        for i, (_, row) in enumerate(data.iterrows(), 1):
            prompt += f"\n【データ{i}】\n"
            prompt += f"テキスト: {row[text_column]}\n"

            for col in metadata_columns:
                if col in data.columns and pd.notna(row[col]):
                    prompt += f"{col}: {row[col]}\n"

        prompt += f"""

上記のデータを基に、以下の形式で分析結果を提供してください:

1. **要約**: データ全体の概要と主要な傾向
2. **主要な発見**: 特に注目すべき点やパターン
3. **詳細分析**: 具体的な分析結果と根拠
4. **推奨事項**: データから導かれる提案や改善点

日本語で詳細に回答してください。
"""

        return prompt


def create_markdown_report(result: AnalysisResult) -> str:
    """分析結果のMarkdownレポートを作成する

    Args:
        result: 分析結果

    Returns:
        Markdownフォーマットのレポート
    """
    metadata_str = (
        ", ".join(result.metadata_columns) if result.metadata_columns else "なし"
    )

    return f"""# テキスト分析結果

## 📝 分析クエリ
{result.query}

## ⚙️ 分析設定
- **プロバイダー**: {result.settings.provider.get_display_name()}
- **モデル**: {result.settings.model.get_display_name()}
- **分析対象件数**: {result.stats["total_count"]}件
- **テキストカラム**: {result.text_column}
- **メタデータカラム**: {metadata_str}

## 📊 基本統計
- **分析対象件数**: {result.stats["total_count"]}
- **平均文字数**: {result.stats["avg_length"]:.0f}
- **ユニークテキスト数**: {result.stats["unique_count"]}
- **最小文字数**: {result.stats["min_length"]}
- **最大文字数**: {result.stats["max_length"]}

## 🤖 分析結果

{result.result}

---
*分析実行日時: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
