"""データ分析関連のPydanticスキーマ定義"""

from enum import Enum

from pydantic import BaseModel


class DataExtractionMethod(Enum):
    """データ抽出方法"""

    KEYWORD_FILTER = "キーワードフィルタ"
    SENTIMENT_FILTER = "感情フィルタ"
    TOPIC_FILTER = "トピックフィルタ"
    CUSTOM_CONDITION = "カスタム条件"


class AnalysisType(Enum):
    """分析タイプ"""

    SUMMARY = "要約"
    INSIGHTS = "インサイト抽出"
    COMPARISON = "比較分析"
    TREND = "トレンド分析"
    CATEGORIZATION = "カテゴリ分析"


class DataExtractionResult(BaseModel):
    """データ抽出結果"""

    method: DataExtractionMethod
    condition: str
    extracted_count: int
    total_count: int
    extracted_indices: list[int]
    summary: str


class AnalysisInstruction(BaseModel):
    """分析指示の構造化"""

    original_instruction: str
    extraction_method: DataExtractionMethod
    extraction_condition: str
    analysis_type: AnalysisType
    specific_requirements: list[str]
    target_columns: list[str]


class AnalysisResult(BaseModel):
    """分析結果"""

    instruction: AnalysisInstruction
    extraction_result: DataExtractionResult
    analysis_summary: str
    key_findings: list[str]
    insights: list[str]
    recommendations: list[str]
    confidence_score: float