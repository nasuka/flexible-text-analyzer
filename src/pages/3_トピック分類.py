import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from schema.llm_providers import LLMModel, LLMProvider
from services.text_column_estimator import (
    estimate_text_column,
    get_text_column_recommendations,
)
from services.topic_classifier import LLMTopicClassifier


def create_classification_charts(df_classified: pd.DataFrame) -> tuple:
    """分類結果の可視化"""

    # トピック分布
    topic_counts = df_classified["main_topic_name"].value_counts()
    fig_topic_dist = px.pie(
        values=topic_counts.values, names=topic_counts.index, title="メイントピック分布"
    )

    # サブトピック分布（上位10）
    subtopic_counts = df_classified["subtopic_name"].value_counts().head(10)
    fig_subtopic_dist = px.bar(
        x=subtopic_counts.values,
        y=subtopic_counts.index,
        orientation="h",
        title="サブトピック分布（上位10）",
        labels={"x": "件数", "y": "サブトピック"},
    )

    # 信頼度分布
    fig_confidence = px.histogram(
        df_classified,
        x="confidence",
        nbins=20,
        title="分類信頼度分布",
        labels={"x": "信頼度", "y": "件数"},
    )

    return fig_topic_dist, fig_subtopic_dist, fig_confidence


def create_topic_subtopic_matrix(df_classified: pd.DataFrame) -> go.Figure:
    """トピックとサブトピックの関係を可視化"""

    crosstab = pd.crosstab(
        df_classified["main_topic_name"], df_classified["subtopic_name"]
    )

    fig = px.imshow(
        crosstab.values,
        labels={"x": "サブトピック", "y": "メイントピック", "color": "件数"},
        x=crosstab.columns,
        y=crosstab.index,
        title="トピックとサブトピックの関係",
    )

    return fig


def main():
    st.title("📊 LLMトピック分類")
    st.markdown("---")

    # LLM API設定
    st.header("🔑 LLM設定")

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
        st.warning(
            f"⚠️ {selected_provider.get_display_name()} API キーを入力してください"
        )
        return

    # データ入力設定
    st.header("📥 データ入力")

    # 入力方法
    input_method = st.radio(
        "データ入力方法を選択してください",
        [
            "トピック分析の結果を使用",
            "ページ上でトピックを定義",
            "JSONファイルをアップロード",
            "CSVファイルとJSONファイルを組み合わせる",
        ],
    )

    topics_data = None
    df = None
    text_column = None

    if input_method == "トピック分析の結果を使用":
        # セッションからトピック分析結果を取得
        if "topics_result" in st.session_state:
            topics_result = st.session_state["topics_result"]

            topics_data = topics_result.dict()
            st.success("✅ トピック分析結果を読み込みました")

            # トピック一覧を表示
            with st.expander("📋 読み込んだトピック一覧"):
                for topic in topics_data["topics"]:
                    st.write(f"**トピック{topic['id']}: {topic['name']}**")
                    for subtopic in topic.get("subtopics", []):
                        st.write(f"  - {subtopic['name']}")

            # CSVファイルのアップロード
            st.subheader("テキストCSVファイル")
            uploaded_file = st.file_uploader(
                "テキストデータを含むCSVファイルをアップロードしてください",
                type=["csv"],
                key="classification_csv",
            )

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ CSVファイルを読み込みました（{len(df)}行）")

                # テキストカラム推定
                recommended_column, analysis = estimate_text_column(df)

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
                                st.metric(
                                    "日本語率", f"{details['japanese_ratio']:.1%}"
                                )
                            with col2:
                                st.metric(
                                    "ユニーク率", f"{details['uniqueness_ratio']:.1%}"
                                )
                            with col3:
                                st.metric("平均文字数", f"{details['avg_length']:.0f}")
                            st.divider()

                # テキスト列の選択（推奨カラムをデフォルトに）
                default_index = 0
                if recommended_column and recommended_column in df.columns:
                    default_index = df.columns.tolist().index(recommended_column)

                text_column = st.selectbox(
                    "テキストを含む列を選択してください",
                    options=df.columns.tolist(),
                    index=default_index,
                )
        else:
            st.warning("⚠️ トピック分析の結果がありません")

    elif input_method == "ページ上でトピックを定義":
        # ページ上でのトピック定義
        st.subheader("📝 トピック定義")

        # サブトピック分類のオプション
        use_subtopics = st.checkbox(
            "サブトピック分類を含める",
            value=False,
            help="チェックするとサブトピックも定義して分類できます",
        )

        # セッション状態の初期化
        if "custom_topics" not in st.session_state:
            st.session_state["custom_topics"] = [
                {
                    "id": 1,
                    "name": "",
                    "description": "",
                    "keywords": "",
                    "subtopics": [],
                }
            ]

        st.markdown("#### トピック追加・編集")

        # トピック追加ボタン
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➕ トピック追加"):
                new_id = (
                    max([t["id"] for t in st.session_state["custom_topics"]], default=0)
                    + 1
                )
                st.session_state["custom_topics"].append(
                    {
                        "id": new_id,
                        "name": "",
                        "description": "",
                        "keywords": "",
                        "subtopics": [],
                    }
                )
                st.rerun()

        with col2:
            if st.button("🗑️ 全てリセット"):
                st.session_state["custom_topics"] = [
                    {
                        "id": 1,
                        "name": "",
                        "description": "",
                        "keywords": "",
                        "subtopics": [],
                    }
                ]
                st.rerun()

        # 各トピックの編集
        topics_to_remove = []

        for i, topic in enumerate(st.session_state["custom_topics"]):
            with st.expander(
                f"トピック {topic['id']}: {topic['name'] or '（未設定）'}",
                expanded=True,
            ):
                col1, col2 = st.columns([4, 1])

                with col1:
                    # トピック基本情報
                    topic["name"] = st.text_input(
                        "トピック名",
                        value=topic["name"],
                        key=f"topic_name_{topic['id']}",
                        placeholder="例: 商品の品質",
                    )

                    topic["description"] = st.text_area(
                        "説明",
                        value=topic["description"],
                        key=f"topic_desc_{topic['id']}",
                        placeholder="例: 商品の品質に関する意見や評価",
                        height=80,
                    )

                    topic["keywords"] = st.text_input(
                        "キーワード（カンマ区切り）",
                        value=topic["keywords"],
                        key=f"topic_keywords_{topic['id']}",
                        placeholder="例: 品質, 質, 良い, 悪い, 耐久性",
                    )

                with col2:
                    if len(st.session_state["custom_topics"]) > 1:
                        if st.button(
                            "🗑️",
                            key=f"remove_topic_{topic['id']}",
                            help="このトピックを削除",
                        ):
                            topics_to_remove.append(i)

                # サブトピック定義
                if use_subtopics:
                    st.markdown("**サブトピック**")

                    # サブトピック追加ボタン
                    if st.button(
                        "➕ サブトピック追加", key=f"add_subtopic_{topic['id']}"
                    ):
                        new_subtopic_id = (
                            max([s["id"] for s in topic["subtopics"]], default=0) + 1
                        )
                        topic["subtopics"].append(
                            {
                                "id": new_subtopic_id,
                                "name": "",
                                "description": "",
                                "keywords": "",
                            }
                        )
                        st.rerun()

                    # サブトピック編集
                    subtopics_to_remove = []
                    for j, subtopic in enumerate(topic["subtopics"]):
                        st.markdown(f"**サブトピック {subtopic['id']}**")

                        col_sub1, col_sub2 = st.columns([4, 1])

                        with col_sub1:
                            subtopic["name"] = st.text_input(
                                "名前",
                                value=subtopic["name"],
                                key=f"subtopic_name_{topic['id']}_{subtopic['id']}",
                                placeholder="例: 耐久性",
                                label_visibility="collapsed",
                            )

                            subtopic["description"] = st.text_area(
                                "説明",
                                value=subtopic["description"],
                                key=f"subtopic_desc_{topic['id']}_{subtopic['id']}",
                                placeholder="例: 商品の耐久性に関する評価",
                                height=60,
                                label_visibility="collapsed",
                            )

                            subtopic["keywords"] = st.text_input(
                                "キーワード",
                                value=subtopic["keywords"],
                                key=f"subtopic_keywords_{topic['id']}_{subtopic['id']}",
                                placeholder="例: 耐久性, 長持ち, 壊れやすい",
                                label_visibility="collapsed",
                            )

                        with col_sub2:
                            if st.button(
                                "🗑️",
                                key=f"remove_subtopic_{topic['id']}_{subtopic['id']}",
                                help="このサブトピックを削除",
                            ):
                                subtopics_to_remove.append(j)

                    # サブトピック削除処理
                    for idx in reversed(subtopics_to_remove):
                        topic["subtopics"].pop(idx)

                    if subtopics_to_remove:
                        st.rerun()

        # トピック削除処理
        for idx in reversed(topics_to_remove):
            st.session_state["custom_topics"].pop(idx)

        if topics_to_remove:
            st.rerun()

        # トピック定義の検証と構造化データ作成
        valid_topics = []
        for topic in st.session_state["custom_topics"]:
            if topic["name"].strip():
                # キーワードをリストに変換
                keywords = [
                    kw.strip() for kw in topic["keywords"].split(",") if kw.strip()
                ]

                valid_topic = {
                    "id": topic["id"],
                    "name": topic["name"].strip(),
                    "description": topic["description"].strip()
                    or topic["name"].strip(),
                    "keywords": keywords,
                }

                if use_subtopics:
                    valid_subtopics = []
                    for subtopic in topic["subtopics"]:
                        if subtopic["name"].strip():
                            sub_keywords = [
                                kw.strip()
                                for kw in subtopic["keywords"].split(",")
                                if kw.strip()
                            ]
                            valid_subtopics.append(
                                {
                                    "id": subtopic["id"],
                                    "name": subtopic["name"].strip(),
                                    "description": subtopic["description"].strip()
                                    or subtopic["name"].strip(),
                                    "keywords": sub_keywords,
                                }
                            )
                    valid_topic["subtopics"] = valid_subtopics
                else:
                    valid_topic["subtopics"] = []

                valid_topics.append(valid_topic)

        if valid_topics:
            topics_data = {"topics": valid_topics}

            # 定義されたトピックのプレビュー
            with st.expander("📋 定義されたトピック一覧"):
                for topic in valid_topics:
                    st.write(f"**トピック{topic['id']}: {topic['name']}**")
                    st.write(f"　説明: {topic['description']}")
                    if topic["keywords"]:
                        st.write(f"　キーワード: {', '.join(topic['keywords'])}")

                    if use_subtopics and topic.get("subtopics"):
                        st.write("　サブトピック:")
                        for subtopic in topic["subtopics"]:
                            st.write(
                                f"　　- {subtopic['name']}: {subtopic['description']}"
                            )
                            if subtopic["keywords"]:
                                st.write(
                                    f"　　　キーワード: {', '.join(subtopic['keywords'])}"
                                )
                    st.divider()

        # CSVファイルのアップロード
        st.subheader("テキストCSVファイル")
        uploaded_file = st.file_uploader(
            "テキストデータを含むCSVファイルをアップロードしてください",
            type=["csv"],
            key="classification_csv_custom",
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSVファイルを読み込みました（{len(df)}行）")

            # テキストカラム推定
            recommended_column, analysis = estimate_text_column(df)

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

            # テキスト列の選択（推奨カラムをデフォルトに）
            default_index = 0
            if recommended_column and recommended_column in df.columns:
                default_index = df.columns.tolist().index(recommended_column)

            text_column = st.selectbox(
                "テキストを含む列を選択してください",
                options=df.columns.tolist(),
                index=default_index,
            )

    elif input_method == "JSONファイルをアップロード":
        st.subheader("トピック定義JSONファイル")
        json_file = st.file_uploader(
            "トピック定義を含むJSONファイルをアップロードしてください",
            type=["json"],
            key="topics_json",
        )

        if json_file is not None:
            try:
                topics_data = json.load(json_file)
                if "topics" in topics_data:
                    topics_data = topics_data["topics"]
                st.success("✅ トピック定義を読み込みました")

                # トピック一覧を表示
                with st.expander("📋 読み込んだトピック一覧"):
                    if isinstance(topics_data, dict) and "topics" in topics_data:
                        for topic in topics_data["topics"]:
                            st.write(f"**トピック{topic['id']}: {topic['name']}**")
                            for subtopic in topic.get("subtopics", []):
                                st.write(f"{subtopic['name']}")
                    elif isinstance(topics_data, list):
                        for topic in topics_data:
                            st.write(f"**トピック{topic['id']}: {topic['name']}**")
                            for subtopic in topic.get("subtopics", []):
                                st.write(f"  - {subtopic['name']}")

            except Exception as e:
                st.error(f"⚠️ JSONファイルの読み込みに失敗しました: {str(e)}")

        # CSVファイルのアップロード
        st.subheader("テキストCSVファイル")
        uploaded_file = st.file_uploader(
            "テキストデータを含むCSVファイルをアップロードしてください",
            type=["csv"],
            key="classification_csv_json",
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ CSVファイルを読み込みました（{len(df)}行）")

            # テキストカラム推定
            recommended_column, analysis = estimate_text_column(df)

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

            # テキスト列の選択（推奨カラムをデフォルトに）
            default_index = 0
            if recommended_column and recommended_column in df.columns:
                default_index = df.columns.tolist().index(recommended_column)

            text_column = st.selectbox(
                "テキストを含む列を選択してください",
                options=df.columns.tolist(),
                index=default_index,
            )

    else:  # CSVファイルとJSONファイルの両方
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("トピック定義JSONファイル")
            json_file = st.file_uploader(
                "トピック定義を含むJSONファイル", type=["json"], key="topics_json_combo"
            )

            if json_file is not None:
                try:
                    topics_data = json.load(json_file)
                    st.success("✅ トピック定義を読み込みました")
                except Exception as e:
                    st.error(f"⚠️ JSONファイルの読み込みに失敗しました: {str(e)}")

        with col2:
            st.subheader("テキストCSVファイル")
            uploaded_file = st.file_uploader(
                "テキストデータを含むCSVファイル",
                type=["csv"],
                key="classification_csv_combo",
            )

            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ CSVファイルを読み込みました（{len(df)}行）")

                # テキストカラム推定
                recommended_column, analysis = estimate_text_column(df)

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
                                st.metric(
                                    "日本語率", f"{details['japanese_ratio']:.1%}"
                                )
                            with col2:
                                st.metric(
                                    "ユニーク率", f"{details['uniqueness_ratio']:.1%}"
                                )
                            with col3:
                                st.metric("平均文字数", f"{details['avg_length']:.0f}")
                            st.divider()

                # テキスト列の選択（推奨カラムをデフォルトに）
                default_index = 0
                if recommended_column and recommended_column in df.columns:
                    default_index = df.columns.tolist().index(recommended_column)

                text_column = st.selectbox(
                    "テキストを含む列を選択してください",
                    options=df.columns.tolist(),
                    index=default_index,
                )

    # 分類処理
    if topics_data is not None and df is not None and text_column is not None:
        st.header("📊 分類")

        # データ説明入力
        st.subheader("📝 データ説明")
        data_description = st.text_area(
            "データの説明（任意）",
            value="",
            placeholder="例: このデータは商品レビューコメントです。ユーザーが商品に対する感想や評価を記載しています。",
            help="データの内容や背景を説明してください。より正確な分類に役立ちます。",
            height=100,
        )

        # データ制限
        data_limit = st.slider(
            "データ件数",
            min_value=1,
            max_value=len(df),
            value=min(100, len(df)),
            help="APIの制限に基づくデータ件数の制限です",
        )

        # データの前処理
        filtered_df = df.head(data_limit)
        filtered_texts = filtered_df[text_column].dropna().astype(str).tolist()

        st.info(f"📝 データ: {len(filtered_texts)}件のテキスト")

        # トークン数予測
        total_chars = sum(len(text) for text in filtered_texts)
        estimated_tokens = total_chars // 3
        st.warning(
            f"📊 予測トークン数: {estimated_tokens:,} tokens（APIの制限に注意してください）"
        )

        # 並列処理設定
        st.subheader("⚙️ 並列処理設定")
        col1, col2 = st.columns(2)

        with col1:
            batch_size = st.slider(
                "バッチサイズ",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="一度に処理するテキスト数（小さいほど安定、大きいほど高速）",
            )

        with col2:
            max_workers = st.slider(
                "並列処理数",
                min_value=1,
                max_value=20,
                value=10,
                help="同時に実行するバッチ数（多すぎるとAPI制限に注意）",
            )

        # 処理予測情報
        total_batches = (len(filtered_texts) + batch_size - 1) // batch_size
        st.info(
            f"📊 処理予測: {total_batches}バッチ（{batch_size}件ずつ）を{max_workers}並列で処理"
        )

        # 分類実行
        if st.button("🚀 トピック分類実行", type="primary"):
            with st.spinner("🤖 LLMによるトピック分類中..."):
                classifier = LLMTopicClassifier(
                    api_key, selected_model.value, batch_size, max_workers
                )

                # 進捗表示
                progress_bar = st.progress(0)
                progress_text = st.empty()

                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    progress_text.text(message)

                # トピック定義の整形
                if isinstance(topics_data, dict) and "topics" in topics_data:
                    classification_result = classifier.classify_texts_parallel(
                        filtered_texts, topics_data, update_progress, data_description
                    )
                else:
                    classification_result = classifier.classify_texts_parallel(
                        filtered_texts,
                        {"topics": topics_data},
                        update_progress,
                        data_description,
                    )

                progress_bar.progress(100)
                progress_text.text("分類完了！")

                if classification_result:
                    st.success("✅ トピック分類が完了しました")

                    # 分類結果の整形
                    classification_data = []
                    for cls in classification_result.classifications:
                        classification_data.append(
                            {
                                "text_index": cls.text_index,
                                "main_topic_id": cls.main_topic_id,
                                "main_topic_name": cls.main_topic_name,
                                "subtopic_id": cls.subtopic_id,
                                "subtopic_name": cls.subtopic_name,
                                "confidence": cls.confidence,
                                "sentiment": cls.sentiment.value,
                            }
                        )

                    classification_df = pd.DataFrame(classification_data)

                    # デバッグ情報を表示
                    st.write("🔍 デバッグ情報:")
                    st.write(f"  - 元データ件数: {len(filtered_df)}")
                    st.write(f"  - 分類結果件数: {len(classification_df)}")
                    st.write(
                        f"  - 分類結果インデックス範囲: {classification_df['text_index'].min()} - {classification_df['text_index'].max()}"
                    )

                    # 結果の結合
                    result_df = filtered_df.copy()
                    result_df = result_df.reset_index(drop=True)

                    # 分類結果がない行のためのデフォルト値を設定
                    result_df["メイントピックID"] = None
                    result_df["メイントピック"] = "未分類"
                    result_df["サブトピックID"] = None
                    result_df["サブトピック"] = "未分類"
                    result_df["分類確度"] = 0.0
                    result_df["センチメント"] = "未分類"

                    # 分類結果をtext_indexに基づいてマージ
                    for _, row in classification_df.iterrows():
                        idx = row["text_index"]
                        if 0 <= idx < len(result_df):
                            result_df.loc[idx, "メイントピックID"] = row[
                                "main_topic_id"
                            ]
                            result_df.loc[idx, "メイントピック"] = row[
                                "main_topic_name"
                            ]
                            result_df.loc[idx, "サブトピックID"] = row["subtopic_id"]
                            result_df.loc[idx, "サブトピック"] = row["subtopic_name"]
                            result_df.loc[idx, "分類確度"] = row["confidence"]
                            result_df.loc[idx, "センチメント"] = row["sentiment"]

                    # 分類統計を表示
                    classified_count = len(
                        result_df[result_df["メイントピック"] != "未分類"]
                    )
                    st.write(f"  - 分類済み件数: {classified_count} / {len(result_df)}")

                    if classified_count < len(result_df):
                        st.warning(
                            f"⚠️ {len(result_df) - classified_count}件が未分類です。バッチサイズや並列処理数を調整してみてください。"
                        )

                    # セッション状態の保存
                    st.session_state["classification_result"] = result_df
                    st.session_state["classification_summary"] = classification_df

                    # 結果表示
                    st.header("📊 分類結果")

                    # メトリクス表示
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("分類件数", len(result_df))
                    with col2:
                        st.metric(
                            "平均確度", f"{classification_df['confidence'].mean():.2f}"
                        )
                    with col3:
                        unique_topics = classification_df["main_topic_name"].nunique()
                        st.metric("ユニークトピック数", unique_topics)

                    # 分類結果テーブル
                    st.subheader("📋 分類結果テーブル")
                    st.dataframe(
                        result_df[
                            ["メイントピック", "サブトピック", "分類確度", text_column]
                        ].head(10),
                        use_container_width=True,
                    )

                    progress_bar.empty()
                    progress_text.empty()

                else:
                    st.error(
                        "⚠️ トピック分類に失敗しました（APIの応答を確認してください）"
                    )
                    progress_bar.empty()
                    progress_text.empty()

    # セッション状態に分類結果がある場合の表示
    if "classification_result" in st.session_state:
        result_df = st.session_state["classification_result"]
        classification_df = st.session_state["classification_summary"]

        # グラフ
        st.header("📊 グラフ")

        # 分類結果の可視化
        fig_topic_dist, fig_subtopic_dist, fig_confidence = (
            create_classification_charts(classification_df)
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_topic_dist, use_container_width=True)
        with col2:
            st.plotly_chart(fig_confidence, use_container_width=True)

        st.plotly_chart(fig_subtopic_dist, use_container_width=True)

        # トピックとサブトピックの関係
        fig_matrix = create_topic_subtopic_matrix(classification_df)
        st.plotly_chart(fig_matrix, use_container_width=True)

        # ダウンロード
        st.header("📥 ダウンロード")

        col1, col2 = st.columns(2)

        with col1:
            # 全体結果CSV
            csv_full = result_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 全体結果 (CSV)",
                data=csv_full,
                file_name="topic_classification_full.csv",
                mime="text/csv",
                key="download_full_csv",
            )

        with col2:
            # 分類結果CSV
            csv_summary = classification_df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="📥 分類結果 (CSV)",
                data=csv_summary,
                file_name="topic_classification_summary.csv",
                mime="text/csv",
                key="download_summary_csv",
            )

        # JSON形式で出力
        classification_json = {
            "classification_settings": {
                "provider": selected_provider.get_display_name(),
                "model": selected_model.get_display_name(),
                "data_count": len(result_df),
                "unique_topics": classification_df["main_topic_name"].nunique(),
            },
            "classifications": classification_df.to_dict("records"),
        }

        json_str = json.dumps(classification_json, ensure_ascii=False, indent=2)

        st.download_button(
            label="📥 分類結果 (JSON)",
            data=json_str,
            file_name="topic_classification.json",
            mime="application/json",
            key="download_classification_json",
        )

    else:
        # 使用方法
        with st.expander("📖 使用方法"):
            st.markdown("""
            ### 📖 使用方法
            1. **OpenAI API Key**の設定
            2. **入力方法**の選択:
               - トピック分析結果から分類
               - JSONファイルをアップロード
               - CSVとJSONファイルの両方
            3. **テキストCSVファイル**のアップロード
            4. **分類実行**の選択
            5. **トピック分類結果**の確認

            ### 🔍 機能説明
            - **トピック分類**: LLMによる自動トピック分類
            - **信頼度表示**: 分類結果の信頼度を表示
            - **可視化**: 結果のグラフ表示
            - **ダウンロード**: 結果をCSV出力

            ### 📊 出力形式
            - **全体結果**: CSV + トピック分類結果
            - **分類結果**: 分類結果の表示
            - **JSON形式**: 全データの保存

            ### ⚙️ 設定項目
            - **API設定**: トークン数の制限設定
            - **モデル選択**: 使用するモデルの選択
            - **バッチ処理**: 50件ずつの一括処理
            - **並列処理**: 複数バッチの同時実行

            ### 💡 技術仕様
            - **Structured Output**: 構造化された出力
            - **バッチ処理**: 効率的な大量データ処理
            - **並列処理**: ThreadPoolExecutorによる高速化
            - **可視化**: 結果のグラフ表示
            - **分類精度**: 高精度な分類処理
            - **拡張性**: 柔軟な機能拡張

            ### 🚀 パフォーマンス改善
            - **バッチサイズ**: 10-100件で調整可能
            - **並列数**: 1-10並列で調整可能
            - **進捗表示**: リアルタイム進捗確認
            - **エラーハンドリング**: 個別バッチのエラー処理
            """)


if __name__ == "__main__":
    main()
