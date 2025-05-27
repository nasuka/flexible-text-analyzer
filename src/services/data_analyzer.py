"""データ分析サービス"""

import pandas as pd
import streamlit as st

from schema.data_analysis import (
    AnalysisInstruction,
    AnalysisResult,
    AnalysisType,
    DataExtractionMethod,
    DataExtractionResult,
)
from services.llm import LLMClient


class LLMDataAnalyzer:
    """LLMを使用したデータ分析サービス"""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """
        データ分析サービスの初期化

        Args:
            api_key: APIキー
            model: 使用するモデル
        """
        self.llm_client = LLMClient(api_key=api_key, model=model)

    def parse_instruction(self, instruction: str, available_columns: list[str]) -> AnalysisInstruction | None:
        """
        自然言語の指示を構造化データに変換

        Args:
            instruction: ユーザーの指示
            available_columns: 利用可能なカラム名のリスト

        Returns:
            構造化された分析指示
        """
        prompt = f"""
ユーザーからの以下の分析指示を解析して、構造化された分析計画を作成してください。

指示: {instruction}

利用可能なカラム: {', '.join(available_columns)}

以下の観点で分析してください：
1. どのようなデータを抽出したいのか（抽出方法と条件）
2. どのような分析を行いたいのか（分析タイプ）
3. 具体的な要求事項は何か
4. どのカラムを対象とするか

抽出方法の選択肢：
- KEYWORD_FILTER: キーワードを含むデータを抽出
- SENTIMENT_FILTER: 特定の感情（ポジティブ、ネガティブ、中立）のデータを抽出
- TOPIC_FILTER: 特定のトピックに関連するデータを抽出
- CUSTOM_CONDITION: その他の複雑な条件

分析タイプの選択肢：
- SUMMARY: 要約生成
- INSIGHTS: インサイト抽出
- COMPARISON: 比較分析
- TREND: トレンド分析
- CATEGORIZATION: カテゴリ分析
"""

        system_message = """
あなたはデータ分析の専門家です。ユーザーの自然言語での指示を、構造化された分析計画に変換してください。
指示が曖昧な場合は、最も合理的な解釈を行ってください。
"""

        try:
            result = self.llm_client.structured_completion(
                prompt=prompt,
                response_format=AnalysisInstruction,
                system_message=system_message,
                temperature=0.3,
            )
            return result
        except Exception as e:
            st.error(f"指示の解析に失敗しました: {str(e)}")
            return None

    def extract_data(
        self, df: pd.DataFrame, instruction: AnalysisInstruction, text_column: str
    ) -> DataExtractionResult | None:
        """
        指示に基づいてデータを抽出

        Args:
            df: データフレーム
            instruction: 分析指示
            text_column: テキストカラム名

        Returns:
            データ抽出結果
        """
        try:
            extracted_indices = []
            total_count = len(df)

            if instruction.extraction_method == DataExtractionMethod.KEYWORD_FILTER:
                # キーワードフィルタ
                condition = instruction.extraction_condition.lower()
                mask = df[text_column].str.lower().str.contains(condition, na=False)
                extracted_indices = df[mask].index.tolist()

            elif instruction.extraction_method == DataExtractionMethod.SENTIMENT_FILTER:
                # 感情フィルタ（感情カラムが存在する場合のみ）
                sentiment_columns = [col for col in df.columns if "sentiment" in col.lower() or "感情" in col]
                if sentiment_columns:
                    sentiment_col = sentiment_columns[0]
                    condition = instruction.extraction_condition.lower()
                    if "ポジティブ" in condition or "positive" in condition:
                        mask = df[sentiment_col].str.contains("ポジティブ|positive", case=False, na=False)
                    elif "ネガティブ" in condition or "negative" in condition:
                        mask = df[sentiment_col].str.contains("ネガティブ|negative", case=False, na=False)
                    else:
                        mask = df[sentiment_col].str.contains("中立|neutral", case=False, na=False)
                    extracted_indices = df[mask].index.tolist()
                else:
                    # 感情カラムがない場合はLLMで感情を判定
                    extracted_indices = self._extract_by_llm_sentiment(df, text_column, instruction.extraction_condition)

            elif instruction.extraction_method == DataExtractionMethod.TOPIC_FILTER:
                # トピックフィルタ
                topic_columns = [col for col in df.columns if "topic" in col.lower() or "トピック" in col]
                if topic_columns:
                    topic_col = topic_columns[0]
                    condition = instruction.extraction_condition
                    mask = df[topic_col].str.contains(condition, case=False, na=False)
                    extracted_indices = df[mask].index.tolist()
                else:
                    # トピックカラムがない場合はキーワードベースで抽出
                    condition = instruction.extraction_condition.lower()
                    mask = df[text_column].str.lower().str.contains(condition, na=False)
                    extracted_indices = df[mask].index.tolist()

            elif instruction.extraction_method == DataExtractionMethod.CUSTOM_CONDITION:
                # カスタム条件（より高度なLLMベースの抽出）
                extracted_indices = self._extract_by_llm_condition(df, text_column, instruction.extraction_condition)

            # 抽出結果の要約を生成
            extracted_count = len(extracted_indices)
            if extracted_count > 0:
                sample_texts = df.loc[extracted_indices, text_column].head(3).tolist()
                summary = f"{extracted_count}件のデータを抽出しました。サンプル: " + "、".join(sample_texts[:2])
            else:
                summary = "条件に該当するデータが見つかりませんでした。"

            return DataExtractionResult(
                method=instruction.extraction_method,
                condition=instruction.extraction_condition,
                extracted_count=extracted_count,
                total_count=total_count,
                extracted_indices=extracted_indices,
                summary=summary,
            )

        except Exception as e:
            st.error(f"データ抽出に失敗しました: {str(e)}")
            return None

    def _extract_by_llm_sentiment(self, df: pd.DataFrame, text_column: str, sentiment_condition: str) -> list[int]:
        """LLMを使用した感情ベースの抽出"""
        # 簡易実装：最初の50件をサンプルとして感情判定
        sample_size = min(50, len(df))
        sample_df = df.head(sample_size)
        
        extracted_indices = []
        
        for idx, text in sample_df[text_column].items():
            prompt = f"""
以下のテキストの感情を判定してください。

テキスト: {text}

条件: {sentiment_condition}

このテキストが条件に合致する場合は「YES」、そうでなければ「NO」で回答してください。
"""
            
            result = self.llm_client.simple_completion(
                prompt=prompt,
                system_message="あなたは感情分析の専門家です。",
                temperature=0.1,
            )
            
            if result and "YES" in result.upper():
                extracted_indices.append(idx)
        
        return extracted_indices

    def _extract_by_llm_condition(self, df: pd.DataFrame, text_column: str, condition: str) -> list[int]:
        """LLMを使用したカスタム条件での抽出"""
        # 簡易実装：最初の50件をサンプルとして条件判定
        sample_size = min(50, len(df))
        sample_df = df.head(sample_size)
        
        extracted_indices = []
        
        for idx, text in sample_df[text_column].items():
            prompt = f"""
以下のテキストが指定された条件に合致するかを判定してください。

テキスト: {text}

条件: {condition}

このテキストが条件に合致する場合は「YES」、そうでなければ「NO」で回答してください。
理由も簡潔に説明してください。
"""
            
            result = self.llm_client.simple_completion(
                prompt=prompt,
                system_message="あなたはデータ分析の専門家です。",
                temperature=0.1,
            )
            
            if result and "YES" in result.upper():
                extracted_indices.append(idx)
        
        return extracted_indices

    def analyze_data(
        self, df: pd.DataFrame, instruction: AnalysisInstruction, extraction_result: DataExtractionResult, text_column: str
    ) -> AnalysisResult | None:
        """
        抽出されたデータに対して分析を実行

        Args:
            df: データフレーム
            instruction: 分析指示
            extraction_result: データ抽出結果
            text_column: テキストカラム名

        Returns:
            分析結果
        """
        try:
            if extraction_result.extracted_count == 0:
                return AnalysisResult(
                    instruction=instruction,
                    extraction_result=extraction_result,
                    analysis_summary="抽出されたデータがないため分析を実行できませんでした。",
                    key_findings=[],
                    insights=[],
                    recommendations=["データ抽出条件を見直してください。"],
                    confidence_score=0.0,
                )

            # 抽出されたデータを取得
            extracted_df = df.loc[extraction_result.extracted_indices]
            extracted_texts = extracted_df[text_column].tolist()

            # テキストを結合（最大トークン数に注意）
            combined_text = "\n".join(extracted_texts[:20])  # 最初の20件のみ使用

            # 分析タイプに応じたプロンプトを生成
            if instruction.analysis_type == AnalysisType.SUMMARY:
                analysis_prompt = f"""
以下のテキストデータを要約してください。

データ件数: {extraction_result.extracted_count}件
抽出条件: {extraction_result.condition}

テキストデータ:
{combined_text}

以下の観点で要約してください：
1. 主要なテーマや内容
2. 頻出するキーワードや話題
3. 全体的な傾向や特徴
4. 注目すべき点
"""

            elif instruction.analysis_type == AnalysisType.INSIGHTS:
                analysis_prompt = f"""
以下のテキストデータからインサイトを抽出してください。

データ件数: {extraction_result.extracted_count}件
抽出条件: {extraction_result.condition}

テキストデータ:
{combined_text}

以下の観点でインサイトを抽出してください：
1. 隠れたパターンや傾向
2. 重要な気づきや発見
3. ビジネス上の示唆
4. アクションにつながる洞察
"""

            else:
                # その他の分析タイプ
                analysis_prompt = f"""
以下のテキストデータを分析してください（分析タイプ: {instruction.analysis_type.value}）。

データ件数: {extraction_result.extracted_count}件
抽出条件: {extraction_result.condition}

テキストデータ:
{combined_text}

詳細な分析結果と実用的な洞察を提供してください。
"""

            # LLMで分析実行
            analysis_result = self.llm_client.simple_completion(
                prompt=analysis_prompt,
                system_message="あなたはデータ分析の専門家です。提供されたデータから価値のある洞察を抽出してください。",
                temperature=0.3,
            )

            if not analysis_result:
                raise Exception("分析結果を取得できませんでした")

            # 結果を構造化
            # 簡易実装：分析結果をパースして構造化
            lines = analysis_result.split("\n")
            key_findings = []
            insights = []
            recommendations = []

            for line in lines:
                line = line.strip()
                if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                    if "発見" in line or "結果" in line:
                        key_findings.append(line.lstrip("•-* "))
                    elif "洞察" in line or "インサイト" in line:
                        insights.append(line.lstrip("•-* "))
                    elif "推奨" in line or "提案" in line:
                        recommendations.append(line.lstrip("•-* "))

            # デフォルト値を設定
            if not key_findings:
                key_findings = ["分析結果の詳細は要約をご確認ください"]
            if not insights:
                insights = ["詳細な洞察は要約をご確認ください"]
            if not recommendations:
                recommendations = ["具体的な推奨事項は要約をご確認ください"]

            # 信頼度スコアを計算（簡易実装）
            confidence_score = min(0.9, extraction_result.extracted_count / 10 * 0.8 + 0.1)

            return AnalysisResult(
                instruction=instruction,
                extraction_result=extraction_result,
                analysis_summary=analysis_result,
                key_findings=key_findings,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score,
            )

        except Exception as e:
            st.error(f"データ分析に失敗しました: {str(e)}")
            return None

    def analyze_with_instruction(
        self, df: pd.DataFrame, instruction: str, text_column: str, available_columns: list[str] | None = None
    ) -> AnalysisResult | None:
        """
        指示文からワンストップで分析を実行

        Args:
            df: データフレーム
            instruction: 分析指示
            text_column: テキストカラム名
            available_columns: 利用可能なカラム（指定なしの場合は全カラム）

        Returns:
            分析結果
        """
        if available_columns is None:
            available_columns = df.columns.tolist()

        # 1. 指示の解析
        parsed_instruction = self.parse_instruction(instruction, available_columns)
        if not parsed_instruction:
            return None

        # 2. データの抽出
        extraction_result = self.extract_data(df, parsed_instruction, text_column)
        if not extraction_result:
            return None

        # 3. データの分析
        analysis_result = self.analyze_data(df, parsed_instruction, extraction_result, text_column)
        return analysis_result