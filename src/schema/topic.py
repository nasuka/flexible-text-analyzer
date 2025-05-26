"""トピック関連のPydanticスキーマ定義"""

from pydantic import BaseModel


class SubTopic(BaseModel):
    """サブトピックのデータモデル"""

    id: int
    name: str
    description: str
    keywords: list[str]


class Topic(BaseModel):
    """トピックのデータモデル"""

    id: int
    name: str
    description: str
    keywords: list[str]
    subtopics: list[SubTopic]


class TopicAnalysisResult(BaseModel):
    """トピック分析結果のデータモデル"""

    topics: list[Topic]
    summary: str


class SentimentAnalysis(BaseModel):
    """感情分析のデータモデル"""

    overall_sentiment: str
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    key_insights: list[str]
