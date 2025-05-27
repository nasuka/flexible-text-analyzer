"""LLMトピック抽出サービス"""

from schema.topic import SentimentAnalysis, TopicAnalysisResult
from services.llm import LLMClient


class LLMTopicExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """共通LLMクライアントを使用したトピック抽出のStructured Output対応"""
        self.llm_client = LLMClient(api_key=api_key, model=model)

    def extract_topics(
        self,
        texts: list[str],
        n_topics: int | None = None,
        n_subtopics: int | None = None,
        data_description: str | None = None,
    ) -> TopicAnalysisResult | None:
        """LLMを使用してトピックとサブトピックを抽出する"""

        # テキストの結合
        combined_text = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(texts)])

        # トピック数とサブトピック数の指定
        topic_instruction = (
            "適切な数のトピック" if n_topics is None else f"最大{n_topics}個のトピック"
        )
        subtopic_instruction = (
            "適切な数のサブトピック"
            if n_subtopics is None
            else f"各トピックにつき最大{n_subtopics}個のサブトピック"
        )

        # データ説明の追加
        data_context = ""
        if data_description and data_description.strip():
            data_context = f"""
データの背景情報:
{data_description.strip()}

この背景情報を考慮して、より適切で文脈に合ったトピック抽出を行ってください。
"""

        prompt = f"""
以下のテキストから{topic_instruction}と、{subtopic_instruction}を抽出してください。

{data_context}
テキスト数: {len(texts)}個:
{combined_text}

指示:
1. テキストの内容を詳細に分析して、自然で意味のあるトピックを抽出してください
2. トピック数は内容に応じて最適な数を自動判定してください（指定がない場合）
3. サブトピックは主トピックの具体的な側面や詳細を表現してください
4. サブトピック数も内容に応じて最適な数を自動判定してください（指定がない場合）
5. キーワードは実際にテキストに出現する重要な単語を選択してください
6. 日本語で自然な表現を使用してください
7. 全体の要約も含めてください
"""

        system_message = "あなたはテキスト分析の専門家です。与えられたテキストから包括的で正確なトピック分析を行ってください。"

        return self.llm_client.structured_completion(
            prompt=prompt,
            response_format=TopicAnalysisResult,
            system_message=system_message,
            temperature=0.3,
        )

    def extract_topics_with_predefined(
        self,
        texts: list[str],
        predefined_topics: list[str],
        n_subtopics: int | None = None,
        data_description: str | None = None,
    ) -> TopicAnalysisResult | None:
        """ユーザー定義トピックを使用してサブトピックを自動生成する"""

        # テキストの結合
        combined_text = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(texts)])

        # サブトピック数の指定
        subtopic_instruction = (
            "適切な数のサブトピック"
            if n_subtopics is None
            else f"各トピックにつき最大{n_subtopics}個のサブトピック"
        )

        # ユーザー定義トピックのリスト
        topics_list = "\n".join([f"{i + 1}. {topic}" for i, topic in enumerate(predefined_topics)])

        # データ説明の追加
        data_context = ""
        if data_description and data_description.strip():
            data_context = f"""
データの背景情報:
{data_description.strip()}

この背景情報を考慮して、より適切で文脈に合ったサブトピック生成を行ってください。
"""

        prompt = f"""
以下のテキストを分析し、指定されたトピックに基づいて{subtopic_instruction}を抽出してください。

指定されたトピック:
{topics_list}

{data_context}
テキスト数: {len(texts)}個:
{combined_text}

指示:
1. **指定されたトピックのみを使用**してください（追加トピックは作成しない）
2. 各トピックに対して、テキスト内容から適切なサブトピックを自動生成してください
3. サブトピックは主トピックの具体的な側面や詳細を表現してください
4. サブトピック数は内容に応じて最適な数を自動判定してください（指定がない場合）
5. キーワードは実際にテキストに出現する重要な単語を選択してください
6. 各トピックに関連する内容がテキストにない場合でも、トピック構造は維持してください
7. 日本語で自然な表現を使用してください
8. 全体の要約も含めてください

重要: トピックIDは1から始まり、指定されたトピックの順序を維持してください。
"""

        system_message = "あなたはテキスト分析の専門家です。ユーザーが指定したトピックに基づいて、適切なサブトピックを自動生成してください。"

        return self.llm_client.structured_completion(
            prompt=prompt,
            response_format=TopicAnalysisResult,
            system_message=system_message,
            temperature=0.3,
        )

    def analyze_sentiment(self, texts: list[str]) -> SentimentAnalysis | None:
        """感情分析を実行する"""

        combined_text = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(texts)])

        prompt = f"""
以下のテキストの感情分析を行ってください。{len(texts)}個のテキストに対して、感情の傾向を分析してください。

テキスト:
{combined_text}

指示::
1. 全体の感情（ポジティブ、ネガティブ、中立）を判定
2. 各感情の割合を0-1で計算
3. 重要な洞察を3-5個抽出
4. 日本語で回答してください
"""

        system_message = (
            "あなたは感情分析の専門家です。テキストの感情を分析してください。"
        )

        return self.llm_client.structured_completion(
            prompt=prompt,
            response_format=SentimentAnalysis,
            system_message=system_message,
            temperature=0.3,
        )
