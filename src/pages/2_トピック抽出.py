import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from schema.llm_providers import LLMModel, LLMProvider
from schema.topic import SentimentAnalysis, TopicAnalysisResult
from services.text_column_estimator import (
    estimate_text_column,
    get_text_column_recommendations,
)
from services.topic_extractor import LLMTopicExtractor


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
            "line": {"width": 2, "color": "white"},
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

    # LLM API設定
    st.header("LLM設定")

    # プロバイダー選択
    provider_display_names = [provider.get_display_name() for provider in LLMProvider]
    selected_provider_display = st.selectbox(
        "LLMプロバイダー",
        provider_display_names,
        help="使用するLLMプロバイダーを選択してください",
    )

    # 選択されたプロバイダーを取得
    selected_provider = None
    for provider in LLMProvider:
        if provider.get_display_name() == selected_provider_display:
            selected_provider = provider
            break

    # APIキー入力
    api_key_label = f"{selected_provider.get_display_name()} API Key"
    api_key_help = f"{selected_provider.get_display_name()} APIキーを入力してください"

    if selected_provider == LLMProvider.OPENAI:
        api_key = st.text_input(
            api_key_label,
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help=api_key_help,
        )
    elif selected_provider == LLMProvider.OPENROUTER:
        api_key = st.text_input(
            api_key_label,
            value=os.getenv("OPENROUTER_API_KEY", ""),
            type="password",
            help=api_key_help,
        )
    else:
        api_key = st.text_input(
            api_key_label,
            type="password",
            help=api_key_help,
        )

    # モデル選択
    available_models = LLMModel.get_models_by_provider(selected_provider)
    model_display_names = [model.get_display_name() for model in available_models]

    selected_model_display = st.selectbox(
        "モデル選択",
        model_display_names,
        help="使用するモデルを選択してください",
    )

    # 選択されたモデルを取得
    selected_model = None
    for model in available_models:
        if model.get_display_name() == selected_model_display:
            selected_model = model
            break

    if not api_key:
        st.warning(f"{selected_provider.get_display_name()} APIキーを入力してください")
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

            # テキストカラム推定
            recommended_column, analysis = estimate_text_column(df)

            # テキスト選択
            st.header("分析設定")

            # 推奨カラム表示
            if recommended_column:
                st.success(f"💡 推奨テキストカラム: **{recommended_column}**")

                with st.expander("📊 カラム分析詳細"):
                    recommendations = get_text_column_recommendations(df, top_n=3)
                    for i, rec in enumerate(recommendations):
                        col_name = rec["column"]
                        details = rec["details"]
                        st.write(
                            f"**{i + 1}位: {col_name}** (スコア: {rec['score']:.1f})"
                        )
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("日本語率", f"{details['japanese_ratio']:.1%}")
                        with col2:
                            st.metric(
                                "ユニーク率", f"{details['uniqueness_ratio']:.1%}"
                            )
                        with col3:
                            st.metric("平均文字数", f"{details['avg_length']:.0f}")
                        st.divider()

            # カラム選択（推奨カラムをデフォルトに）
            default_index = 0
            if recommended_column and recommended_column in df.columns:
                default_index = df.columns.tolist().index(recommended_column)

            text_column = st.selectbox(
                "テキスト列を選択してください",
                options=df.columns.tolist(),
                index=default_index,
                help="トピック抽出したいテキストデータを含む列を選択してください",
            )

            if text_column:
                # トピック定義方法の選択
                st.subheader("トピック定義方法")
                extraction_method = st.radio(
                    "トピック定義方法を選択",
                    ["完全自動", "ユーザー定義トピック"],
                    index=0,
                    help="完全自動：LLMがデータから自動でトピックとサブトピックを決定\nユーザー定義：指定したトピックから自動でサブトピックを生成",
                )

                user_topics = None
                if extraction_method == "ユーザー定義トピック":
                    st.subheader("トピック定義")

                    # セッション状態でユーザー定義トピックを管理
                    if "user_defined_topics" not in st.session_state:
                        st.session_state.user_defined_topics = [""]

                    st.write(
                        "分析したいトピックを入力してください。サブトピックは自動で生成されます。"
                    )

                    # トピック入力フィールド
                    topics_container = st.container()
                    with topics_container:
                        for i, topic in enumerate(st.session_state.user_defined_topics):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                new_topic = st.text_input(
                                    f"トピック {i + 1}",
                                    value=topic,
                                    key=f"user_topic_{i}",
                                    placeholder="例: 商品の品質について",
                                )
                                st.session_state.user_defined_topics[i] = new_topic
                            with col2:
                                if st.button(
                                    "削除",
                                    key=f"delete_topic_{i}",
                                    disabled=len(st.session_state.user_defined_topics)
                                    <= 1,
                                ):
                                    st.session_state.user_defined_topics.pop(i)
                                    st.rerun()

                    # トピック追加ボタン
                    if st.button(
                        "トピック追加",
                        disabled=len(st.session_state.user_defined_topics) >= 8,
                    ):
                        st.session_state.user_defined_topics.append("")
                        st.rerun()

                    # 空でないトピックのみを取得
                    user_topics = [
                        topic.strip()
                        for topic in st.session_state.user_defined_topics
                        if topic.strip()
                    ]

                    if user_topics:
                        st.success(f"定義されたトピック: {len(user_topics)}個")
                        for i, topic in enumerate(user_topics, 1):
                            st.write(f"{i}. {topic}")
                    else:
                        st.warning("少なくとも1つのトピックを入力してください")

                # パラメータ設定
                st.subheader("分析パラメータ")
                col1, col2 = st.columns(2)
                with col1:
                    if extraction_method == "完全自動":
                        auto_topics = st.checkbox("トピック数を自動決定", value=True)
                        if not auto_topics:
                            n_topics = st.slider(
                                "トピック数", min_value=2, max_value=10, value=5
                            )
                        else:
                            n_topics = None
                    else:
                        n_topics = len(user_topics) if user_topics else None
                        st.info(
                            f"トピック数: {n_topics if n_topics else 0}個（ユーザー定義）"
                        )

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

                # データ説明の入力
                st.subheader("データ説明（オプション）")
                data_description = st.text_area(
                    "データの説明を入力してください",
                    placeholder="例: YouTubeの商品レビューコメント。主にスマートフォンに関する顧客の評価や意見が含まれています。",
                    help="データの内容や背景を説明することで、より適切なトピック抽出が可能になります",
                    height=80,
                )

                # テキストデータ抽出
                filtered_texts = (
                    df[text_column].dropna().astype(str).tolist()[:data_limit]
                )
                st.info(f"分析データ: {len(filtered_texts)}テキスト")

                # トークン数予測
                total_chars = sum(len(text) for text in filtered_texts)
                estimated_tokens = total_chars // 3
                st.warning(
                    f"予測トークン数: {estimated_tokens:,} tokens（API制限に注意してください）"
                )

                # 分析実行
                if st.button("LLMトピック抽出実行", type="primary"):
                    # バリデーション
                    if len(filtered_texts) < 5:
                        st.error("分析には最低5件のデータが必要です")
                    elif (
                        extraction_method == "ユーザー定義トピック" and not user_topics
                    ):
                        st.error("ユーザー定義トピックが入力されていません")
                    else:
                        with st.spinner("LLMによる分析中..."):
                            extractor = LLMTopicExtractor(api_key, selected_model.value)

                            # トピック抽出
                            st.write("トピック抽出中...")
                            progress_bar = st.progress(0)

                            # ユーザー定義トピックの場合はuser_topicsを渡す
                            if extraction_method == "ユーザー定義トピック":
                                topics_result = (
                                    extractor.extract_topics_with_predefined(
                                        filtered_texts,
                                        user_topics,
                                        n_subtopics,
                                        data_description,
                                    )
                                )
                            else:
                                topics_result = extractor.extract_topics(
                                    filtered_texts,
                                    n_topics,
                                    n_subtopics,
                                    data_description,
                                )
                            progress_bar.progress(50)

                            if topics_result:
                                # セッション状態に結果を保存
                                st.session_state["topics_result"] = topics_result
                                st.session_state["analysis_settings"] = {
                                    "provider": selected_provider.get_display_name(),
                                    "model": selected_model.get_display_name(),
                                    "extraction_method": extraction_method,
                                    "n_topics": n_topics if n_topics else "自動決定",
                                    "n_subtopics": n_subtopics
                                    if n_subtopics
                                    else "自動決定",
                                    "data_count": len(filtered_texts),
                                    "text_column": text_column,
                                    "data_description": data_description
                                    if data_description.strip()
                                    else "なし",
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

                            else:
                                st.error(
                                    "トピック抽出に失敗しました。APIキーを確認してください"
                                )

        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {str(e)}")

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
            "sentiment": sentiment_result.dict() if sentiment_result else None,
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
        # 使用方法の説明
        with st.expander("使用方法・特徴"):
            st.markdown("""
            ### 使用方法
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
            - **推奨モデル**: gpt-4oを推奨

            ### 主な機能
            - **トピック抽出**: テキストから主要なトピックを抽出
            - **構造化出力**: 階層的な分析結果を提供
            - **可視化**: グラフによる直感的な理解
            - **感情分析**: テキストの感情傾向を分析
            """)


if __name__ == "__main__":
    main()
