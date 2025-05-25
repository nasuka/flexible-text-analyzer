import pandas as pd
import streamlit as st
import openai
import json
import time
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import os
from pydantic import BaseModel


class SubTopic(BaseModel):
    """サブトピックのデータモデル"""

    id: int
    name: str
    description: str
    keywords: List[str]


class Topic(BaseModel):
    """トピックのデータモデル"""

    id: int
    name: str
    description: str
    keywords: List[str]
    subtopics: List[SubTopic]


class TopicAnalysisResult(BaseModel):
    """トピック分析結果のデータモデル"""

    topics: List[Topic]
    summary: str


class SentimentAnalysis(BaseModel):
    """感情分析のデータモデル"""

    overall_sentiment: str
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    key_insights: List[str]


class LLMTopicExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06"):
        """OpenAI APIを使用したトピック抽出のStructured Output対応"""
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def extract_topics(
        self, texts: List[str], n_topics: int = 5, n_subtopics: int = 3
    ) -> Optional[TopicAnalysisResult]:
        """LLMを使用してトピックとサブトピックを抽出する"""

        # テキストの結合
        combined_text = "\n".join([f"{i + 1}. {text}" for i, text in enumerate(texts)])

        prompt = f"""
以下のテキストから最大{n_topics}個のトピックと、各トピックにつき最大{n_subtopics}個のサブトピックを抽出してください。

テキスト数: {len(texts)}個:
{combined_text}

指示::
1. テキストを分析してトピックを抽出してください
2. サブトピックを適切に分類してください
3. サブトピックは具体的な内容を含めてください
4. キーワードは関連性の高いものを選んでください
5. 日本語で回答してください
6. 構造化された形式で出力してください
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

    def analyze_sentiment(self, texts: List[str]) -> Optional[SentimentAnalysis]:
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


def create_topic_visualization(result: TopicAnalysisResult) -> go.Figure:
    """トピックの可視化"""
    if not result or not result.topics:
        return None

    # トピックごとのデータを準備
    topic_names = [f"トピック{t.id}: {t.name}" for t in result.topics]
    keyword_counts = [len(t.keywords) for t in result.topics]
    subtopic_counts = [len(t.subtopics) for t in result.topics]

    fig = go.Figure(
        data=[
            go.Bar(
                name="キーワード数",
                x=topic_names,
                y=keyword_counts,
                yaxis="y",
                offsetgroup=1,
            ),
            go.Bar(
                name="サブトピック数",
                x=topic_names,
                y=subtopic_counts,
                yaxis="y2",
                offsetgroup=2,
            ),
        ]
    )

    fig.update_layout(
        title="トピック分析",
        xaxis_title="トピック",
        yaxis={"title": "キーワード数", "side": "left"},
        yaxis2={"title": "サブトピック数", "side": "right", "overlaying": "y"},
        barmode="group",
        height=500,
    )

    return fig


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


def create_topic_network(result: TopicAnalysisResult) -> go.Figure:
    """トピックとサブトピックのネットワーク"""
    if not result or not result.topics:
        return None

    try:
        import networkx as nx
    except ImportError:
        st.warning("NetworkXがインストールされていません。インストールしてください。")
        return None

    G = nx.Graph()

    # ノードの追加
    for topic in result.topics:
        G.add_node(f"T{topic.id}", label=topic.name, type="topic", size=20)
        for subtopic in topic.subtopics:
            G.add_node(
                f"T{topic.id}S{subtopic.id}",
                label=subtopic.name,
                type="subtopic",
                size=10,
            )
            G.add_edge(f"T{topic.id}", f"T{topic.id}S{subtopic.id}")

    # レイアウトの計算
    pos = nx.spring_layout(G, k=2, iterations=50)

    # エッジの描画
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line={"width": 1, "color": "#888"},
        hoverinfo="none",
        mode="lines",
    )

    # ノードの描画
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]["label"])
        node_color.append("red" if G.nodes[node]["type"] == "topic" else "blue")
        node_size.append(G.nodes[node]["size"])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=node_text,
        textposition="middle center",
        marker={
            "size": node_size, 
            "color": node_color, 
            "line": {"width": 2, "color": "white"}
        },
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout={
            "title": "トピックマップ",
            "showlegend": False,
            "hovermode": "closest",
            "margin": {"b": 20, "l": 5, "r": 5, "t": 40},
            "annotations": [
                {
                    "text": "赤: トピック, 青: サブトピック",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.005,
                    "y": -0.002,
                }
            ],
            "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
            "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
        },
    )

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
                    n_topics = st.slider(
                        "トピック数", min_value=2, max_value=10, value=5
                    )
                    n_subtopics = st.slider(
                        "サブトピック数", min_value=1, max_value=5, value=3
                    )

                with col2:
                    include_sentiment = st.checkbox("感情分析を含める", value=True)
                    data_limit = st.slider(
                        "データ件数",
                        min_value=10,
                        max_value=len(df),
                        value=min(100, len(df)),
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

                                # トピックグラフ
                                fig_topics = create_topic_visualization(topics_result)
                                if fig_topics:
                                    st.plotly_chart(
                                        fig_topics, use_container_width=True
                                    )

                                # トピックネットワーク
                                fig_network = create_topic_network(topics_result)
                                if fig_network:
                                    st.plotly_chart(
                                        fig_network, use_container_width=True
                                    )

                                # 感情分析
                                if include_sentiment:
                                    st.write("感情分析中...")
                                    sentiment_result = extractor.analyze_sentiment(
                                        filtered_texts
                                    )
                                    progress_bar.progress(100)

                                    if sentiment_result:
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

                                # レポート出力
                                st.subheader("レポート出力")

                                # 構造化JSON出力
                                download_data = {
                                    "analysis_settings": {
                                        "model": model,
                                        "n_topics": n_topics,
                                        "n_subtopics": n_subtopics,
                                        "data_count": len(filtered_texts),
                                        "text_column": text_column,
                                    },
                                    "topics": topics_result.dict(),
                                    "sentiment": sentiment_result.dict()
                                    if include_sentiment and sentiment_result
                                    else None,
                                }

                                json_str = json.dumps(
                                    download_data, ensure_ascii=False, indent=2
                                )

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.download_button(
                                        label="分析レポート (JSON)",
                                        data=json_str,
                                        file_name="structured_topic_analysis.json",
                                        mime="application/json",
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
                                                    [st.name for st in topic.subtopics]
                                                ),
                                            }
                                        )

                                    csv_df = pd.DataFrame(csv_data)
                                    csv_str = csv_df.to_csv(
                                        index=False, encoding="utf-8-sig"
                                    )

                                    st.download_button(
                                        label="トピック一覧 (CSV)",
                                        data=csv_str,
                                        file_name="topic_summary.csv",
                                        mime="text/csv",
                                    )

                                progress_bar.empty()

                            else:
                                st.error(
                                    "トピック抽出に失敗しました。APIキーを確認してください"
                                )
                                progress_bar.empty()

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
