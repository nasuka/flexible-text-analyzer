"""LLMトピック抽出サービス"""

import openai
import streamlit as st

from schema.topic import SentimentAnalysis, TopicAnalysisResult


class LLMTopicExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """OpenAI APIを使用したトピック抽出のStructured Output対応"""
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def extract_topics(
        self,
        texts: list[str],
        n_topics: int | None = None,
        n_subtopics: int | None = None,
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

        prompt = f"""
以下のテキストから{topic_instruction}と、{subtopic_instruction}を抽出してください。

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

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたはテキスト分析の専門家です。与えられたテキストから包括的で正確なトピック分析を行ってください。",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=TopicAnalysisResult,
                temperature=0.3,
            )

            return response.choices[0].message.parsed

        except Exception as e:
            st.error(f"トピック抽出でエラーが発生しました: {str(e)}")
            return None

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

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは感情分析の専門家です。テキストの感情を分析してください。",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=SentimentAnalysis,
                temperature=0.3,
            )

            return response.choices[0].message.parsed

        except Exception as e:
            st.error(f"感情分析でエラーが発生しました: {str(e)}")
            return None
