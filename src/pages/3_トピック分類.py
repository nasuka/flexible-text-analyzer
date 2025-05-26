import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel


class TopicClassification(BaseModel):
    """トピック分類の結果"""

    text_index: int
    main_topic_id: int
    main_topic_name: str
    subtopic_id: int
    subtopic_name: str
    confidence: float
    reasoning: str


class ClassificationResult(BaseModel):
    """分類結果"""

    classifications: list[TopicClassification]


class LLMTopicClassifier:
    def __init__(self, api_key: str, model: str = "gpt-4o-2024-08-06", batch_size: int = 50, max_workers: int = 5):
        """OpenAI APIを使用してトピック分類のStructured Outputを取得"""
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_workers = max_workers

    def _create_topic_definitions(self, topics_data: dict[str, Any]) -> str:
        """トピック定義の作成"""
        topic_info = []
        for topic in topics_data.get("topics", []):
            topic_str = f"トピック{topic['id']}: {topic['name']}\n"
            topic_str += f"  説明: {topic['description']}\n"
            topic_str += f"  キーワード: {', '.join(topic['keywords'])}\n"

            if topic.get("subtopics"):
                topic_str += "  サブトピック:\n"
                for subtopic in topic["subtopics"]:
                    topic_str += f"    {subtopic['id']}: {subtopic['name']}\n"
                    topic_str += f"      説明: {subtopic['description']}\n"
                    topic_str += (
                        f"      キーワード: {', '.join(subtopic['keywords'])}\n"
                    )

            topic_info.append(topic_str)

        return "\n\n".join(topic_info)

    def _classify_batch(
        self, batch_texts: list[str], batch_start_index: int, topic_definitions: str
    ) -> ClassificationResult | None:
        """バッチでテキストを分類する"""
        
        # テキストリストの作成（バッチ内のインデックスを使用）
        text_list = "\n".join([f"{i}: {text}" for i, text in enumerate(batch_texts)])

        prompt = f"""
以下のトピック定義に基づいて、テキストをトピックとサブトピックに分類してください。

トピック定義:
{topic_definitions}

テキストリスト:
{text_list}

分類ルール:
1. テキストを0から始まるインデックスで指定してください（このバッチ内でのインデックス）
2. メイントピックとサブトピックを指定してください
3. 信頼度を0-1の数値で指定してください
4. 分類理由を簡潔に説明してください
5. 最も適切なトピックを選択してください
"""

        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "あなたはテキストをトピックに分類する専門家です。与えられたトピック定義に基づいてテキストを分類してください。",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=ClassificationResult,
                temperature=0.1,
            )

            result = response.choices[0].message.parsed
            
            # バッチ内のインデックスを全体のインデックスに調整
            if result:
                for classification in result.classifications:
                    classification.text_index += batch_start_index
            
            return result

        except Exception as e:
            st.error(f"バッチ分類でエラーが発生しました: {str(e)}")
            return None

    def classify_texts_parallel(
        self, texts: list[str], topics_data: dict[str, Any], progress_callback=None
    ) -> ClassificationResult | None:
        """テキストを並列でバッチ分類する"""
        
        topic_definitions = self._create_topic_definitions(topics_data)
        
        # テキストをバッチに分割
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batches.append((batch_texts, i))
        
        all_classifications = []
        completed_batches = 0
        
        # 並列処理でバッチを処理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 全バッチのタスクを投入
            future_to_batch = {
                executor.submit(self._classify_batch, batch_texts, start_index, topic_definitions): (batch_texts, start_index)
                for batch_texts, start_index in batches
            }
            
            # 完了したタスクから結果を取得
            for future in as_completed(future_to_batch):
                batch_texts, start_index = future_to_batch[future]
                try:
                    result = future.result()
                    if result and result.classifications:
                        all_classifications.extend(result.classifications)
                    
                    completed_batches += 1
                    
                    # 進捗更新
                    if progress_callback:
                        progress = int((completed_batches / len(batches)) * 100)
                        progress_callback(progress, f"バッチ {completed_batches}/{len(batches)} 完了")
                        
                except Exception as e:
                    st.error(f"バッチ処理でエラーが発生しました: {str(e)}")
        
        # 結果をインデックス順にソート
        all_classifications.sort(key=lambda x: x.text_index)
        
        return ClassificationResult(classifications=all_classifications)

    def classify_texts(
        self, texts: list[str], topics_data: dict[str, Any]
    ) -> ClassificationResult | None:
        """後方互換性のための従来メソッド（非推奨）"""
        return self.classify_texts_parallel(texts, topics_data)


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

    # OpenAI API設定
    st.header("🔑 API設定")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="OpenAI APIキーを入力してください",
    )

    model = st.selectbox(
        "モデル",
        ["gpt-4o-2024-08-06", "gpt-4o-mini"],
        help="Structured Outputに対応したモデルを選択してください",
    )

    if not api_key:
        st.warning("⚠️ OpenAI API キーを入力してください")
        return

    # データ入力設定
    st.header("📥 データ入力")

    # 入力方法
    input_method = st.radio(
        "データ入力方法を選択してください",
        [
            "トピック分析の結果を使用",
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

                # テキスト列の選択
                text_column = st.selectbox(
                    "テキストを含む列を選択してください", options=df.columns.tolist()
                )
        else:
            st.warning("⚠️ トピック分析の結果がありません")

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

            # テキスト列の選択
            text_column = st.selectbox(
                "テキストを含む列を選択してください", options=df.columns.tolist()
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

                # テキスト列の選択
                text_column = st.selectbox(
                    "テキストを含む列を選択してください", options=df.columns.tolist()
                )

    # 分類処理
    if topics_data is not None and df is not None and text_column is not None:
        st.header("📊 分類")

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
                min_value=10,
                max_value=100,
                value=50,
                step=10,
                help="一度に処理するテキスト数"
            )
        
        with col2:
            max_workers = st.slider(
                "並列処理数",
                min_value=1,
                max_value=10,
                value=5,
                help="同時に実行するバッチ数"
            )

        # 処理予測情報
        total_batches = (len(filtered_texts) + batch_size - 1) // batch_size
        st.info(f"📊 処理予測: {total_batches}バッチ（{batch_size}件ずつ）を{max_workers}並列で処理")

        # 分類実行
        if st.button("🚀 トピック分類実行", type="primary"):
            with st.spinner("🤖 LLMによるトピック分類中..."):
                classifier = LLMTopicClassifier(api_key, model, batch_size, max_workers)

                # 進捗表示
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    progress_text.text(message)

                # トピック定義の整形
                if isinstance(topics_data, dict) and "topics" in topics_data:
                    classification_result = classifier.classify_texts_parallel(
                        filtered_texts, topics_data, update_progress
                    )
                else:
                    classification_result = classifier.classify_texts_parallel(
                        filtered_texts, {"topics": topics_data}, update_progress
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
                                "reasoning": cls.reasoning,
                            }
                        )

                    classification_df = pd.DataFrame(classification_data)

                    # 結果の結合
                    result_df = filtered_df.copy()
                    result_df = result_df.reset_index(drop=True)

                    # 分類結果の追加
                    result_df["メイントピックID"] = classification_df["main_topic_id"]
                    result_df["メイントピック"] = classification_df["main_topic_name"]
                    result_df["サブトピックID"] = classification_df["subtopic_id"]
                    result_df["サブトピック"] = classification_df["subtopic_name"]
                    result_df["分類確度"] = classification_df["confidence"]
                    result_df["分類理由"] = classification_df["reasoning"]

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
                "model": model,
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
