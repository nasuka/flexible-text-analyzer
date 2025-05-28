import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from schema.llm_providers import LLMModel, LLMProvider
from services.text_analyzer import (
    AnalysisSettings,
    TextAnalyzer,
    create_markdown_report,
)
from services.text_column_estimator import (
    estimate_text_column,
    get_text_column_recommendations,
)


def filter_data_by_query(
    df: pd.DataFrame, text_column: str, metadata_columns: list[str], query: str
) -> pd.DataFrame:
    """自然言語クエリに基づいてデータをフィルタリング"""
    # 簡単なキーワードベースフィルタリング
    # より高度な実装では、LLMを使用して条件を解釈することも可能

    if not query.strip():
        return df

    # キーワードを抽出
    keywords = [word.strip().lower() for word in query.split() if len(word.strip()) > 1]

    if not keywords:
        return df

    # フィルタリング条件を構築
    mask = pd.Series([False] * len(df))

    for keyword in keywords:
        # テキストカラムでの検索
        text_mask = df[text_column].str.contains(keyword, case=False, na=False)

        # メタデータカラムでの検索
        meta_mask = pd.Series([False] * len(df))
        for col in metadata_columns:
            if col in df.columns:
                meta_mask |= (
                    df[col].astype(str).str.contains(keyword, case=False, na=False)
                )

        mask |= text_mask | meta_mask

    return df[mask]


def create_summary_visualization(summary_data: dict) -> tuple[go.Figure, go.Figure]:
    """要約結果の可視化"""

    # キーワード分布
    if "keywords" in summary_data:
        keywords = summary_data["keywords"][:10]  # 上位10個
        fig_keywords = px.bar(
            x=[kw["count"] for kw in keywords],
            y=[kw["word"] for kw in keywords],
            orientation="h",
            title="主要キーワード（上位10）",
            labels={"x": "出現回数", "y": "キーワード"},
        )
    else:
        fig_keywords = go.Figure()
        fig_keywords.add_annotation(
            text="キーワードデータがありません",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    # 感情分布
    if "sentiment_distribution" in summary_data:
        sentiment_dist = summary_data["sentiment_distribution"]
        fig_sentiment = px.pie(
            values=list(sentiment_dist.values()),
            names=list(sentiment_dist.keys()),
            title="感情分布",
        )
    else:
        fig_sentiment = go.Figure()
        fig_sentiment.add_annotation(
            text="感情データがありません",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )

    return fig_keywords, fig_sentiment


def main():
    st.title("🔍 テキスト分析")
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

    # データ入力
    st.header("📥 データ入力")

    # CSVファイルのアップロード
    uploaded_file = st.file_uploader(
        "テキストデータを含むCSVファイルをアップロードしてください",
        type=["csv"],
        key="analysis_csv",
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
                    st.write(f"**{i + 1}位: {col_name}** (スコア: {rec['score']:.1f})")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("日本語率", f"{details['japanese_ratio']:.1%}")
                    with col2:
                        st.metric("ユニーク率", f"{details['uniqueness_ratio']:.1%}")
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

        # メタデータカラムの選択（複数選択可能）
        other_columns = [col for col in df.columns if col != text_column]
        metadata_columns = st.multiselect(
            "メタデータ列を選択してください（分析に含めるデータ）",
            options=other_columns,
            default=[],
            help="トピック、カテゴリ、感情など、分析に含めたい追加情報を選択してください",
        )

        # 分析クエリ入力
        st.header("🔍 分析指示")

        analysis_query = st.text_area(
            "分析したい内容を自然言語で入力してください",
            value="",
            placeholder="""例:
- ポジティブなコメントを要約して分析して
- 商品の品質に関する意見をまとめて
- 否定的な感情を含むテキストを抽出して要約
- 特定のトピックについての意見をまとめて""",
            height=120,
            help="具体的な分析指示を入力してください。キーワード、感情、トピックなどを指定できます。",
        )

        if analysis_query.strip():
            # 全データを使用（フィルタリングなし）
            st.subheader("📋 データ設定")

            st.info(f"📝 全データを分析対象とします: {len(df)}件")

            # データプレビュー
            with st.expander("📋 データプレビュー"):
                display_columns = [text_column] + metadata_columns
                st.dataframe(
                    df[display_columns].head(10),
                    use_container_width=True,
                )

            # データ制限
            data_limit = st.slider(
                "分析対象データ件数",
                min_value=1,
                max_value=len(df),
                value=min(100, len(df)),
                help="LLMで分析する件数を設定してください（多すぎるとAPI制限に注意）",
            )

            # 分析対象データの準備
            analysis_df = df.head(data_limit)

            # トークン数予測（仮の分析サービスでトークン数推定）
            temp_settings = AnalysisSettings(
                provider=selected_provider,
                model=selected_model,
                api_key="dummy",  # トークン数推定のためのダミー
            )
            temp_analyzer = TextAnalyzer(temp_settings)
            estimated_tokens = temp_analyzer.estimate_tokens(
                analysis_df, text_column, metadata_columns
            )
            st.warning(
                f"📊 予測トークン数: {estimated_tokens:,} tokens（APIの制限に注意してください）"
            )

            # 分析実行
            if st.button("🚀 分析実行", type="primary"):
                with st.spinner("🤖 LLMによるテキスト分析中..."):
                    # 分析設定の作成
                    settings = AnalysisSettings(
                        provider=selected_provider,
                        model=selected_model,
                        api_key=api_key,
                        temperature=0.3,
                        max_tokens=3000,
                    )

                    # テキスト分析サービスの初期化
                    analyzer = TextAnalyzer(settings)

                    # 分析実行
                    result = analyzer.analyze_text(
                        data=df,
                        text_column=text_column,
                        metadata_columns=metadata_columns,
                        query=analysis_query,
                        data_limit=data_limit,
                    )

                    if result:
                        st.success("✅ 分析が完了しました")

                        # 分析結果の表示
                        st.header("📊 分析結果")

                        # 基本統計
                        st.subheader("📈 基本統計")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("分析対象件数", result.stats["total_count"])
                        with col2:
                            st.metric("平均文字数", f"{result.stats['avg_length']:.0f}")
                        with col3:
                            st.metric(
                                "ユニークテキスト数", result.stats["unique_count"]
                            )
                        with col4:
                            st.metric("最大文字数", result.stats["max_length"])

                        # LLM分析結果
                        st.subheader("🤖 LLM分析結果")
                        st.markdown(result.result)

                        # データ詳細
                        st.subheader("📋 分析対象データ詳細")
                        display_columns = [result.text_column] + result.metadata_columns
                        st.dataframe(
                            result.data[display_columns],
                            use_container_width=True,
                        )

                        # セッション状態に保存
                        st.session_state["analysis_result"] = {
                            "query": result.query,
                            "result": result.result,
                            "data": result.data,
                            "text_column": result.text_column,
                            "metadata_columns": result.metadata_columns,
                            "stats": result.stats,
                            "settings": {
                                "provider": result.settings.provider.get_display_name(),
                                "model": result.settings.model.get_display_name(),
                                "data_count": result.stats["total_count"],
                            },
                        }

                        # ダウンロードセクション
                        st.header("📥 ダウンロード")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # 分析対象データCSV
                            csv_data = result.data.to_csv(
                                index=False, encoding="utf-8-sig"
                            )
                            st.download_button(
                                label="📥 分析対象データ (CSV)",
                                data=csv_data,
                                file_name="analysis_data.csv",
                                mime="text/csv",
                                key="download_analysis_data",
                            )

                        with col2:
                            # 分析結果JSON
                            result_json = {
                                "query": result.query,
                                "analysis_result": result.result,
                                "stats": result.stats,
                                "settings": {
                                    "provider": result.settings.provider.get_display_name(),
                                    "model": result.settings.model.get_display_name(),
                                    "data_count": result.stats["total_count"],
                                    "text_column": result.text_column,
                                    "metadata_columns": result.metadata_columns,
                                },
                                "data": result.data.to_dict("records"),
                            }

                            json_str = json.dumps(
                                result_json, ensure_ascii=False, indent=2
                            )
                            st.download_button(
                                label="📥 分析結果 (JSON)",
                                data=json_str,
                                file_name="analysis_result.json",
                                mime="application/json",
                                key="download_analysis_result",
                            )

                        with col3:
                            # 分析結果Markdown
                            markdown_content = create_markdown_report(result)
                            st.download_button(
                                label="📥 分析結果 (Markdown)",
                                data=markdown_content,
                                file_name="analysis_result.md",
                                mime="text/markdown",
                                key="download_analysis_markdown",
                            )

                    else:
                        st.error("⚠️ 分析に失敗しました（APIの応答を確認してください）")

    # セッション状態に分析結果がある場合の再表示
    if "analysis_result" in st.session_state:
        saved_result = st.session_state["analysis_result"]

        st.header("📊 前回の分析結果")
        st.info(f"分析クエリ: {saved_result['query']}")

        with st.expander("🤖 分析結果を再表示"):
            st.markdown(saved_result["result"])

    # 使用方法
    with st.expander("📖 使用方法"):
        st.markdown("""
        ### 📖 使用方法
        1. **LLM API Key**の設定
        2. **CSVファイル**のアップロード
        3. **テキストカラム**の選択
        4. **メタデータカラム**の選択（任意）
        5. **分析指示**を自然言語で入力
        6. **分析実行**で結果を確認

        ### 🔍 分析指示の例
        - "ポジティブなコメントを要約して分析して"
        - "商品の品質に関する意見をまとめて"
        - "否定的な感情を含むテキストを抽出して要約"
        - "特定のキーワードを含むテキストを分析"

        ### 📊 出力内容
        - **要約**: データ全体の概要
        - **主要な発見**: 注目すべき点
        - **詳細分析**: 具体的な分析結果
        - **推奨事項**: 改善提案

        ### 💡 活用シーン
        - **顧客レビュー分析**: 商品やサービスへの意見抽出
        - **ソーシャルメディア分析**: 投稿内容の傾向把握
        - **アンケート分析**: 自由記述回答の要約
        - **コメント分析**: ユーザーの声の整理

        ### ⚙️ 技術仕様
        - **LLM統合**: マルチプロバイダー対応
        - **自動フィルタリング**: キーワードベース抽出
        - **メタデータ活用**: 構造化情報の組み込み
        - **結果出力**: CSV/JSON形式対応
        """)


if __name__ == "__main__":
    main()
