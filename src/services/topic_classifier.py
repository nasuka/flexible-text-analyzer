"""LLMトピック分類サービス"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import streamlit as st

from schema.classification import ClassificationResult
from services.llm import LLMClient


class LLMTopicClassifier:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        batch_size: int = 25,
        max_workers: int = 3,
    ):
        """共通LLMクライアントを使用してトピック分類のStructured Outputを取得"""
        self.llm_client = LLMClient(api_key=api_key, model=model)
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
        self,
        batch_texts: list[str],
        batch_start_index: int,
        topic_definitions: str,
        data_description: str = "",
    ) -> ClassificationResult | None:
        """バッチでテキストを分類する"""
        # テキストリストの作成（バッチ内のインデックスを使用）
        text_list = "\n".join([f"{i}: {text}" for i, text in enumerate(batch_texts)])

        # データ説明部分の構築
        data_context = ""
        if data_description.strip():
            data_context = f"""
データの背景・説明:
{data_description.strip()}

"""

        prompt = f"""
以下のトピック定義に基づいて、テキストをトピックとサブトピックに分類してください。

{data_context}トピック定義:
{topic_definitions}

テキストリスト（{len(batch_texts)}件）:
{text_list}

分類ルール:
1. **必須**: 全ての{len(batch_texts)}件のテキスト（インデックス0から{len(batch_texts) - 1}まで）を必ず分類してください
2. テキストインデックスは0から{len(batch_texts) - 1}の範囲で指定してください
3. メイントピックとサブトピックを指定してください
4. 信頼度を0-1の数値で指定してください
5. 分類理由を簡潔に説明してください
6. 最も適切なトピックを選択してください
7. 分類できないテキストがある場合は、「その他」というトピック・サブトピックに割り当ててください
8. データの背景・説明を参考にして、文脈に適した分類を行ってください

重要: 必ず{len(batch_texts)}件全ての分類結果を返してください。
"""

        system_message = "あなたはテキストをトピックに分類する専門家です。与えられた全てのテキストを必ず分類してください。分類結果の数は入力テキスト数と完全に一致する必要があります。"

        result = self.llm_client.structured_completion(
            prompt=prompt,
            response_format=ClassificationResult,
            system_message=system_message,
            temperature=0.1,
        )

        try:
            # 結果の検証
            if result and result.classifications:
                # 分類結果の数をチェック
                if len(result.classifications) != len(batch_texts):
                    st.warning(
                        f"⚠️ バッチ分類で期待件数と異なる結果: 期待{len(batch_texts)}件、実際{len(result.classifications)}件"
                    )

                # バッチ内のインデックスを全体のインデックスに調整
                for classification in result.classifications:
                    classification.text_index += batch_start_index

                # インデックスの重複チェック
                indices = [cls.text_index for cls in result.classifications]
                if len(set(indices)) != len(indices):
                    st.warning("⚠️ バッチ分類でインデックスの重複が発生しました")

                return result
            else:
                st.error(
                    f"バッチ分類で空の結果が返されました（バッチサイズ: {len(batch_texts)}）"
                )
                return None

        except Exception as e:
            st.error(f"バッチ分類でエラーが発生しました: {str(e)}")
            return None

    def classify_texts_parallel(
        self,
        texts: list[str],
        topics_data: dict[str, Any],
        progress_callback=None,
        data_description: str = "",
    ) -> ClassificationResult | None:
        """テキストを並列でバッチ分類する"""

        topic_definitions = self._create_topic_definitions(topics_data)

        # テキストをバッチに分割
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batches.append((batch_texts, i))

        all_classifications = []
        completed_batches = 0
        failed_batches = []

        # 並列処理でバッチを処理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 全バッチのタスクを投入
            future_to_batch = {
                executor.submit(
                    self._classify_batch,
                    batch_texts,
                    start_index,
                    topic_definitions,
                    data_description,
                ): (batch_texts, start_index)
                for batch_texts, start_index in batches
            }

            # 完了したタスクから結果を取得
            for future in as_completed(future_to_batch):
                batch_texts, start_index = future_to_batch[future]
                try:
                    result = future.result()
                    if result and result.classifications:
                        all_classifications.extend(result.classifications)
                        if progress_callback:
                            progress_callback(
                                int((completed_batches + 1) / len(batches) * 100),
                                f"バッチ {completed_batches + 1}/{len(batches)} 完了 ({len(result.classifications)}件分類)",
                            )
                    else:
                        failed_batches.append((start_index, len(batch_texts)))
                        if progress_callback:
                            progress_callback(
                                int((completed_batches + 1) / len(batches) * 100),
                                f"バッチ {completed_batches + 1}/{len(batches)} 失敗",
                            )

                    completed_batches += 1

                except Exception as e:
                    failed_batches.append((start_index, len(batch_texts)))
                    st.error(
                        f"バッチ処理でエラーが発生しました（開始インデックス: {start_index}）: {str(e)}"
                    )
                    completed_batches += 1

        # 失敗したバッチの情報を表示
        if failed_batches:
            st.warning(f"⚠️ {len(failed_batches)}個のバッチで分類に失敗しました:")
            for start_idx, batch_size in failed_batches:
                st.write(f"  - インデックス {start_idx} から {batch_size} 件")

        # 結果をインデックス順にソート
        all_classifications.sort(key=lambda x: x.text_index)

        # 分類結果の統計
        expected_total = len(texts)
        actual_total = len(all_classifications)

        if progress_callback:
            progress_callback(100, f"完了: {actual_total}/{expected_total} 件分類")

        st.info(
            f"📊 分類完了: {actual_total}/{expected_total} 件 ({actual_total / expected_total * 100:.1f}%)"
        )

        return ClassificationResult(classifications=all_classifications)

    def classify_texts(
        self, texts: list[str], topics_data: dict[str, Any], data_description: str = ""
    ) -> ClassificationResult | None:
        """後方互換性のための従来メソッド（非推奨）"""
        return self.classify_texts_parallel(
            texts, topics_data, data_description=data_description
        )
