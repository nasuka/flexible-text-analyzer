import json
import os

import openai
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
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


class LLMTopicExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
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
                        "content": "あなたはテキスト分析の専門家です。与えられたテキストからトピックを抽出してください。",
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



def create_sentiment_chart(sentiment: SentimentAnalysis) -> go.Figure:
    """感情分析の可視化"""
    if not sentiment:
        return None

    labels = ["ポジティブ", "中立", "ネガティブ"]
    values = [
        sentiment.positive_ratio,
        sentiment.neutral_ratio,
        sentiment.negative_ratio,
    ]
    colors = ["#00CC96", "#FFA15A", "#EF553B"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo="label+percent",
                hole=0.3,
            )
        ]
    )

    fig.update_layout(title="感情分析", height=400)

    return fig



def main():
    st.title("LLMによるトピック抽出 (Structured Output)")
    st.markdown("---")

    # OpenAI API設定
    st.header("API設定")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="OpenAI APIキーを入力してください",
    )

    model = st.selectbox(
        "モデル選択",
        ["gpt-4o-2024-08-06", "gpt-4o-mini"],
        help="Structured Output対応のモデルを選択してください",
    )

    if not api_key:
        st.warning("OpenAI APIキーを入力してください")
        return

    # CSVファイルアップロード
    st.header("データ入力")
    uploaded_file = st.file_uploader(
        "CSVファイルをアップロードしてください",
        type=["csv"],
        help="テキストデータを含むCSVファイルをアップロードしてください",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"ファイル読み込み成功: {len(df)}行, {len(df.columns)}列")

            # データプレビュー
            with st.expander("データプレビュー"):
                st.dataframe(df.head(), use_container_width=True)

            # テキスト選択
            st.header("分析設定")
            text_column = st.selectbox(
                "テキスト列を選択してください",
                options=df.columns.tolist(),
                help="トピック抽出したいテキストデータを含む列を選択してください",
            )

            if text_column:
                # パラメータ設定
                col1, col2 = st.columns(2)
                with col1:
                    auto_topics = st.checkbox("トピック数を自動決定", value=True)
                    if not auto_topics:
                        n_topics = st.slider(
                            "トピック数", min_value=2, max_value=10, value=5
                        )
                    else:
                        n_topics = None

                    auto_subtopics = st.checkbox("サブトピック数を自動決定", value=True)
                    if not auto_subtopics:
                        n_subtopics = st.slider(
                            "サブトピック数", min_value=1, max_value=5, value=3
                        )
                    else:
                        n_subtopics = None

                with col2:
                    include_sentiment = st.checkbox("感情分析を含める", value=True)
                    data_limit = st.slider(
                        "データ件数",
                        min_value=10,
                        max_value=len(df),
                        value=len(df),
                    )

                # テキストデータ抽出
                filtered_texts = (
                    df[text_column].dropna().astype(str).tolist()[:data_limit]
                )
                st.info(f"分析データ: {len(filtered_texts)}テキスト")

                # トークン数予測
                total_chars = sum(len(text) for text in filtered_texts)
                estimated_tokens = total_chars // 3  # 概算
                st.warning(
                    f"予測トークン数: {estimated_tokens:,} tokens (API制限に注意)"
                )

                # 分析実行
                if st.button("LLMトピック抽出実行", type="primary"):
                    if len(filtered_texts) < 5:
                        st.error("分析には最低5件のデータが必要です")
                    else:
                        with st.spinner("LLMによる分析中..."):
                            extractor = LLMTopicExtractor(api_key, model)

                            # トピック抽出
                            st.write("トピック抽出中...")
                            progress_bar = st.progress(0)

                            topics_result = extractor.extract_topics(
                                filtered_texts, n_topics, n_subtopics
                            )
                            progress_bar.progress(50)

                            if topics_result:
                                # セッション状態に結果を保存
                                st.session_state["topics_result"] = topics_result
                                st.session_state["analysis_settings"] = {
                                    "model": model,
                                    "n_topics": n_topics if n_topics else "自動決定",
                                    "n_subtopics": n_subtopics
                                    if n_subtopics
                                    else "自動決定",
                                    "data_count": len(filtered_texts),
                                    "text_column": text_column,
                                }
                                st.success("トピック抽出完了")

                                # レポート
                                st.header("分析レポート")

                                # 概要
                                st.subheader("全体概要")
                                st.write(topics_result.summary)

                                # トピック一覧
                                st.subheader("トピック一覧")

                                for topic in topics_result.topics:
                                    with st.expander(
                                        f"トピック{topic.id}: {topic.name}"
                                    ):
                                        st.write(f"**説明:** {topic.description}")
                                        st.write(
                                            f"**キーワード:** {', '.join(topic.keywords)}"
                                        )

                                        if topic.subtopics:
                                            st.write("**サブトピック:**")
                                            for subtopic in topic.subtopics:
                                                st.write(
                                                    f"  **{subtopic.name}**: {subtopic.description}"
                                                )
                                                st.write(
                                                    f"    キーワード: {', '.join(subtopic.keywords)}"
                                                )

                                # 可視化
                                st.subheader("可視化")


                                # 感情分析
                                if include_sentiment:
                                    st.write("感情分析中...")
                                    sentiment_result = extractor.analyze_sentiment(
                                        filtered_texts
                                    )
                                    progress_bar.progress(100)

                                    if sentiment_result:
                                        # セッション状態に感情分析結果も保存
                                        st.session_state["sentiment_result"] = (
                                            sentiment_result
                                        )

                                        st.subheader("感情分析レポート")

                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric(
                                                "全体感情",
                                                sentiment_result.overall_sentiment,
                                            )
                                            st.metric(
                                                "ポジティブ率",
                                                f"{sentiment_result.positive_ratio:.1%}",
                                            )
                                            st.metric(
                                                "ネガティブ率",
                                                f"{sentiment_result.negative_ratio:.1%}",
                                            )

                                        with col2:
                                            fig_sentiment = create_sentiment_chart(
                                                sentiment_result
                                            )
                                            if fig_sentiment:
                                                st.plotly_chart(
                                                    fig_sentiment,
                                                    use_container_width=True,
                                                )

                                        # 洞察
                                        st.write("**主要な洞察:**")
                                        for insight in sentiment_result.key_insights:
                                            st.write(f"• {insight}")

                                progress_bar.empty()

                # セッション状態から結果を表示（ダウンロード後もセッションがリセットされない）
                if "topics_result" in st.session_state:
                    topics_result = st.session_state["topics_result"]
                    analysis_settings = st.session_state.get("analysis_settings", {})
                    sentiment_result = st.session_state.get("sentiment_result", None)

                    # レポート出力
                    st.subheader("レポート出力")

                    # 構造化JSON出力
                    download_data = {
                        "analysis_settings": analysis_settings,
                        "topics": topics_result.dict(),
                        "sentiment": sentiment_result.dict()
                        if sentiment_result
                        else None,
                    }

                    json_str = json.dumps(download_data, ensure_ascii=False, indent=2)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="分析レポート (JSON)",
                            data=json_str,
                            file_name="structured_topic_analysis.json",
                            mime="application/json",
                            key="download_json",
                        )

                    with col2:
                        # CSV形式トピック一覧
                        csv_data = []
                        for topic in topics_result.topics:
                            csv_data.append(
                                {
                                    "トピックID": topic.id,
                                    "トピック名": topic.name,
                                    "説明": topic.description,
                                    "キーワード": ", ".join(topic.keywords),
                                    "サブトピック数": len(topic.subtopics),
                                    "サブトピック一覧": "; ".join(
                                        [subtopic.name for subtopic in topic.subtopics]
                                    ),
                                }
                            )

                        csv_df = pd.DataFrame(csv_data)
                        csv_str = csv_df.to_csv(index=False, encoding="utf-8-sig")

                        st.download_button(
                            label="トピック一覧 (CSV)",
                            data=csv_str,
                            file_name="topic_summary.csv",
                            mime="text/csv",
                            key="download_csv",
                        )

                else:
                    st.error("トピック抽出に失敗しました。APIキーを確認してください")

        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {str(e)}")

    else:
        st.info("CSVファイルをアップロードしてください")

        # 使い方説明
        with st.expander("使い方ガイド"):
            st.markdown("""
            ### 使い方
            1. **OpenAI API Key**を[取得](https://platform.openai.com/api-keys)して入力
            2. **Structured Output対応モデル**を選択
            3. **CSVファイル**をアップロード
            4. **テキスト列**を選択
            5. **パラメータ**を設定
            6. **LLMトピック抽出実行**をクリック

            ### Structured Outputの特徴
            - **高精度**: 構造化された出力で分析
            - **詳細な分析**: トピックとサブトピックの階層構造
            - **効率的**: 自動的にデータを整理
            - **高速処理**: バッチ処理による効率化

            ### サンプルCSV形式
            ```csv
            id,comment,author,date
            1,素晴らしい商品でした,ユーザー1,2024-01-01
            2,改善が必要な点があります,ユーザー2,2024-01-02
            3,期待以上の品質です,ユーザー3,2024-01-03
            ```

            ### 注意事項
            - **API制限**: 大量のデータを分析する場合は注意
            - **処理時間**: データ量に応じて時間がかかります
            - **トークン制限**: モデルの制限に注意
            - **推奨モデル**: gpt-4o-2024-08-06を推奨

            ### 主な機能
            - **トピック抽出**: テキストから主要なトピックを抽出
            - **構造化出力**: 階層的な分析結果を提供
            - **可視化**: グラフによる直感的な理解
            - **感情分析**: テキストの感情傾向を分析
            """)


if __name__ == "__main__":
    main()
