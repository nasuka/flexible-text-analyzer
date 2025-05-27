import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from schema.data_analysis import AnalysisResult, DataExtractionMethod
from schema.llm_providers import LLMModel, LLMProvider
from services.data_analyzer import LLMDataAnalyzer
from services.text_column_estimator import (
    estimate_text_column,
    get_text_column_recommendations,
)


def create_extraction_chart(extraction_result) -> go.Figure:
    """データ抽出結果の可視化"""
    labels = ["抽出データ", "その他"]
    values = [extraction_result.extracted_count, extraction_result.total_count - extraction_result.extracted_count]
    colors = ["#00CC96", "#FFA15A"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker_colors=colors,
                textinfo="label+percent+value",
                hole=0.3,
            )
        ]
    )

    fig.update_layout(
        title=f"データ抽出結果 ({extraction_result.method.value})",
        height=400,
    )

    return fig


def create_confidence_gauge(confidence_score: float) -> go.Figure:
    """信頼度ゲージの作成"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence_score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "分析信頼度"},
            delta={"reference": 80},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 80], "color": "yellow"},
                    {"range": [80, 100], "color": "green"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=400)
    return fig


def main():
    st.title("🤖 テキスト指示によるデータ分析")
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
        st.warning(f"{selected_provider.get_display_name()} APIキーを入力してください")
        return

    # CSVファイルアップロード
    st.header("📁 データ入力")
    uploaded_file = st.file_uploader(
        "CSVファイルをアップロードしてください",
        type=["csv"],
        help="分析したいテキストデータを含むCSVファイルをアップロードしてください",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"ファイル読み込み成功: {len(df)}行, {len(df.columns)}列")

            # データプレビュー
            with st.expander("📊 データプレビュー"):
                st.dataframe(df.head(), use_container_width=True)

            # テキストカラム推定
            recommended_column, analysis = estimate_text_column(df)

            # テキスト選択
            st.header("⚙️ 分析設定")

            # 推奨カラム表示
            if recommended_column:
                st.success(f"💡 推奨テキストカラム: **{recommended_column}**")

                with st.expander("📈 カラム分析詳細"):
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

            # カラム選択（推奨カラムをデフォルトに）
            default_index = 0
            if recommended_column and recommended_column in df.columns:
                default_index = df.columns.tolist().index(recommended_column)

            text_column = st.selectbox(
                "テキスト列を選択してください",
                options=df.columns.tolist(),
                index=default_index,
                help="分析したいテキストデータを含む列を選択してください",
            )

            # 利用可能なメタデータカラムの検出
            metadata_columns = []
            topic_columns = [col for col in df.columns if "topic" in col.lower() or "トピック" in col]
            sentiment_columns = [col for col in df.columns if "sentiment" in col.lower() or "感情" in col]
            classification_columns = [col for col in df.columns if "classification" in col.lower() or "分類" in col]

            if topic_columns:
                metadata_columns.extend(topic_columns)
            if sentiment_columns:
                metadata_columns.extend(sentiment_columns)
            if classification_columns:
                metadata_columns.extend(classification_columns)

            if metadata_columns:
                st.info(f"🏷️ メタデータカラムを検出しました: {', '.join(metadata_columns)}")
                st.write("これらのカラムを使用してより精密なデータ抽出が可能です。")

            # 分析指示の入力
            st.header("💬 分析指示")
            
            # サンプル指示の表示
            with st.expander("💡 サンプル指示"):
                st.markdown("""
                **基本的な分析指示の例:**
                - "ポジティブなコメントを要約して分析して"
                - "商品の品質について言及している内容を分析"
                - "ネガティブな意見の主要な問題点を抽出"
                - "価格に関するコメントの傾向を分析"
                - "改善提案を含むフィードバックを分類"
                
                **高度な分析指示の例:**
                - "満足度の高いレビューから成功要因を分析"
                - "苦情コメントから改善すべき点を特定"
                - "競合他社との比較言及を抽出して分析"
                - "リピート購入に関する言及を分析"
                """)

            analysis_instruction = st.text_area(
                "分析したい内容を自然な言葉で入力してください",
                placeholder="例: ポジティブなコメントを要約して、どのような点が評価されているか分析してください",
                help="「○○について分析して」「××を要約して」などの自然な指示を入力してください。AIが指示を解釈してデータを抽出・分析します。",
                height=100,
            )

            if analysis_instruction.strip():
                # データ制限設定
                col1, col2 = st.columns(2)
                with col1:
                    data_limit = st.slider(
                        "分析データ件数",
                        min_value=10,
                        max_value=min(1000, len(df)),
                        value=min(500, len(df)),
                        help="処理するデータの最大件数（API制限とコストを考慮してください）",
                    )

                with col2:
                    # トークン数予測
                    sample_texts = df[text_column].dropna().head(data_limit)
                    total_chars = sum(len(str(text)) for text in sample_texts)
                    estimated_tokens = total_chars // 3
                    st.metric("予測トークン数", f"{estimated_tokens:,}")

                # 分析実行
                if st.button("🚀 分析実行", type="primary"):
                    if len(df) < 5:
                        st.error("分析には最低5件のデータが必要です")
                    else:
                        with st.spinner("🤖 AIが指示を解析し、データを分析中..."):
                            # データアナライザーの初期化
                            analyzer = LLMDataAnalyzer(api_key, selected_model.value)

                            # 分析対象データを制限
                            analysis_df = df.head(data_limit)

                            # 分析実行
                            progress_bar = st.progress(0)
                            st.write("📋 指示を解析中...")
                            progress_bar.progress(20)

                            analysis_result = analyzer.analyze_with_instruction(
                                analysis_df, analysis_instruction, text_column, df.columns.tolist()
                            )

                            progress_bar.progress(100)

                            if analysis_result:
                                # セッション状態に結果を保存
                                st.session_state["analysis_result"] = analysis_result
                                st.session_state["analysis_settings"] = {
                                    "provider": selected_provider.get_display_name(),
                                    "model": selected_model.get_display_name(),
                                    "instruction": analysis_instruction,
                                    "text_column": text_column,
                                    "data_count": len(analysis_df),
                                    "metadata_columns": metadata_columns,
                                }

                                st.success("✅ 分析完了！")

                                # 結果表示
                                st.header("📊 分析結果")

                                # 概要
                                st.subheader("📋 分析概要")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "抽出データ件数",
                                        analysis_result.extraction_result.extracted_count,
                                    )
                                with col2:
                                    st.metric(
                                        "抽出率",
                                        f"{analysis_result.extraction_result.extracted_count / analysis_result.extraction_result.total_count:.1%}",
                                    )
                                with col3:
                                    st.metric(
                                        "信頼度",
                                        f"{analysis_result.confidence_score:.1%}",
                                    )

                                # 解析された指示の表示
                                with st.expander("🔍 AI解析結果: 指示の構造化"):
                                    instruction = analysis_result.instruction
                                    st.write(f"**元の指示:** {instruction.original_instruction}")
                                    st.write(f"**抽出方法:** {instruction.extraction_method.value}")
                                    st.write(f"**抽出条件:** {instruction.extraction_condition}")
                                    st.write(f"**分析タイプ:** {instruction.analysis_type.value}")
                                    if instruction.specific_requirements:
                                        st.write(f"**具体的要件:** {', '.join(instruction.specific_requirements)}")

                                # データ抽出結果
                                st.subheader("📈 データ抽出結果")
                                col1, col2 = st.columns([2, 1])
                                with col1:
                                    st.write(analysis_result.extraction_result.summary)

                                with col2:
                                    fig_extraction = create_extraction_chart(analysis_result.extraction_result)
                                    st.plotly_chart(fig_extraction, use_container_width=True)

                                # 分析結果
                                st.subheader("🎯 分析結果")
                                st.write(analysis_result.analysis_summary)

                                # 主要な発見
                                if analysis_result.key_findings:
                                    st.subheader("🔍 主要な発見")
                                    for finding in analysis_result.key_findings:
                                        st.write(f"• {finding}")

                                # インサイト
                                if analysis_result.insights:
                                    st.subheader("💡 インサイト")
                                    for insight in analysis_result.insights:
                                        st.write(f"• {insight}")

                                # 推奨事項
                                if analysis_result.recommendations:
                                    st.subheader("📝 推奨事項")
                                    for recommendation in analysis_result.recommendations:
                                        st.write(f"• {recommendation}")

                                # 信頼度ゲージ
                                col1, col2 = st.columns([1, 1])
                                with col2:
                                    fig_confidence = create_confidence_gauge(analysis_result.confidence_score)
                                    st.plotly_chart(fig_confidence, use_container_width=True)

                                progress_bar.empty()

                            else:
                                st.error("❌ 分析に失敗しました。APIキーや指示内容を確認してください")

        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {str(e)}")

    # セッション状態から結果を表示（ダウンロード後もセッションがリセットされない）
    if "analysis_result" in st.session_state:
        analysis_result: AnalysisResult = st.session_state["analysis_result"]
        analysis_settings = st.session_state.get("analysis_settings", {})

        # 結果出力
        st.subheader("💾 結果出力")

        # 構造化JSON出力
        download_data = {
            "analysis_settings": analysis_settings,
            "instruction": analysis_result.instruction.dict(),
            "extraction_result": analysis_result.extraction_result.dict(),
            "analysis_summary": analysis_result.analysis_summary,
            "key_findings": analysis_result.key_findings,
            "insights": analysis_result.insights,
            "recommendations": analysis_result.recommendations,
            "confidence_score": analysis_result.confidence_score,
        }

        json_str = json.dumps(download_data, ensure_ascii=False, indent=2)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📄 分析レポート (JSON)",
                data=json_str,
                file_name="data_analysis_result.json",
                mime="application/json",
                key="download_json",
            )

        with col2:
            # 簡易CSV出力
            csv_data = {
                "指示": [analysis_result.instruction.original_instruction],
                "抽出方法": [analysis_result.instruction.extraction_method.value],
                "抽出条件": [analysis_result.instruction.extraction_condition],
                "分析タイプ": [analysis_result.instruction.analysis_type.value],
                "抽出件数": [analysis_result.extraction_result.extracted_count],
                "総件数": [analysis_result.extraction_result.total_count],
                "信頼度": [f"{analysis_result.confidence_score:.1%}"],
                "主要発見": ["; ".join(analysis_result.key_findings)],
                "インサイト": ["; ".join(analysis_result.insights)],
                "推奨事項": ["; ".join(analysis_result.recommendations)],
            }

            csv_df = pd.DataFrame(csv_data)
            csv_str = csv_df.to_csv(index=False, encoding="utf-8-sig")

            st.download_button(
                label="📊 分析サマリー (CSV)",
                data=csv_str,
                file_name="analysis_summary.csv",
                mime="text/csv",
                key="download_csv",
            )

    else:
        # 使用方法の説明
        with st.expander("📖 使用方法・特徴"):
            st.markdown("""
            ### 🎯 この機能について
            自然言語での指示を入力するだけで、AIが自動的にデータを抽出・分析します。

            ### 📝 使用方法
            1. **APIキー設定**: OpenAI APIキーまたは対応プロバイダーのキーを入力
            2. **データアップロード**: 分析したいCSVファイルをアップロード
            3. **テキスト列選択**: 分析対象のテキストカラムを選択
            4. **分析指示入力**: 自然な言葉で分析したい内容を入力
            5. **分析実行**: AIが指示を解釈してデータを分析

            ### 🚀 主な機能
            - **自然言語理解**: 「ポジティブなコメントを分析」などの抽象的な指示に対応
            - **自動データ抽出**: 指示に基づいて関連データを自動抽出
            - **高度な分析**: LLMによる詳細な分析とインサイト生成
            - **メタデータ活用**: トピックや感情分析結果を活用した精密な抽出

            ### 💡 指示の例
            - **要約系**: "ポジティブなコメントを要約"
            - **分析系**: "商品の品質について言及している内容を分析"
            - **抽出系**: "改善提案を含むフィードバックを特定"
            - **比較系**: "満足度の高いレビューと低いレビューを比較"

            ### ⚠️ 注意事項
            - **API制限**: 大量データの処理時はAPI制限に注意
            - **処理時間**: データ量に応じて処理時間が増加します
            - **精度**: AIの解釈のため、複雑な指示は段階的に実行を推奨
            - **コスト**: トークン使用量に応じてAPI利用料金が発生します

            ### 🔧 対応データ形式
            - **CSV**: UTF-8エンコーディング推奨
            - **テキストカラム**: 日本語・英語対応
            - **メタデータ**: トピック、感情、分類カラムを自動検出
            """)


if __name__ == "__main__":
    main()