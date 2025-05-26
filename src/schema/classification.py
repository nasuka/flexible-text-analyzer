"""分類関連のPydanticスキーマ定義"""

from enum import Enum

from pydantic import BaseModel


class Sentiment(Enum):
    POSITIVE = "ポジティブ"
    NEGATIVE = "ネガティブ"
    NEUTRAL = "中立"


class TopicClassification(BaseModel):
    """トピック分類の結果"""

    text_index: int
    main_topic_id: int
    main_topic_name: str
    subtopic_id: int
    subtopic_name: str
    confidence: float
    sentiment: Sentiment


class ClassificationResult(BaseModel):
    """分類結果"""

    classifications: list[TopicClassification]
