import re

import pandas as pd


def estimate_text_column(df: pd.DataFrame) -> tuple[str | None, dict]:
    """
    CSVデータフレームから分析に適したテキストカラムを推定する

    Args:
        df: pandas DataFrame

    Returns:
        tuple: (推定されたカラム名, 分析詳細情報)

    分析基準:
    1. 日本語の文字が多く含まれている
    2. ユニークな値が多い（多様性がある）
    3. 文字列データである
    4. 空値が少ない
    """
    if df.empty:
        return None, {"error": "データフレームが空です"}

    column_scores = {}
    analysis_details = {}

    for column in df.columns:
        try:
            # 文字列でない列はスキップ
            series = df[column].astype(str)

            # 基本統計
            total_count = len(series)
            non_null_count = series.dropna().count()
            unique_count = series.nunique()

            # 空値が多すぎる場合はスキップ
            if non_null_count / total_count < 0.3:
                continue

            # 日本語文字の割合を計算
            japanese_char_ratio = calculate_japanese_ratio(series)

            # ユニークさの計算（重複の少なさ）
            uniqueness_ratio = (
                unique_count / non_null_count if non_null_count > 0 else 0
            )

            # 平均文字数
            avg_length = series.str.len().mean()

            # 文字列の長さの分散（多様性）
            length_variance = series.str.len().var()

            # スコア計算（重み付け）
            score = 0

            # 日本語文字の割合（重要度：高）
            score += japanese_char_ratio * 40

            # ユニークさの割合（重要度：高）
            score += uniqueness_ratio * 30

            # 平均文字数（適度な長さが好ましい）
            if 10 <= avg_length <= 500:
                score += 20
            elif 5 <= avg_length < 10 or 500 < avg_length <= 1000:
                score += 10

            # 長さの多様性（重要度：中）
            if length_variance and length_variance > 100:
                score += 10

            column_scores[column] = score
            analysis_details[column] = {
                "japanese_ratio": japanese_char_ratio,
                "uniqueness_ratio": uniqueness_ratio,
                "avg_length": avg_length,
                "length_variance": length_variance,
                "non_null_ratio": non_null_count / total_count,
                "unique_count": unique_count,
                "score": score,
            }

        except Exception as e:
            analysis_details[column] = {"error": str(e)}
            continue

    if not column_scores:
        return None, {
            "error": "分析に適したテキストカラムが見つかりませんでした",
            "details": analysis_details,
        }

    # 最高スコアのカラムを選択
    best_column = max(column_scores, key=column_scores.get)

    return best_column, {
        "recommended_column": best_column,
        "all_scores": column_scores,
        "analysis_details": analysis_details,
    }


def calculate_japanese_ratio(series: pd.Series) -> float:
    """
    pandas Seriesの日本語文字の割合を計算

    Args:
        series: pandas Series（文字列）

    Returns:
        float: 日本語文字の割合（0.0-1.0）
    """
    total_chars = 0
    japanese_chars = 0

    # 日本語文字のパターン（ひらがな、カタカナ、漢字）
    japanese_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]")

    for text in series.dropna():
        text_str = str(text)
        total_chars += len(text_str)
        japanese_chars += len(japanese_pattern.findall(text_str))

    if total_chars == 0:
        return 0.0

    return japanese_chars / total_chars


def get_text_column_recommendations(df: pd.DataFrame, top_n: int = 3) -> list:
    """
    テキストカラムの推奨順位を取得

    Args:
        df: pandas DataFrame
        top_n: 上位何位まで返すか

    Returns:
        list: 推奨カラムのリスト（スコア順）
    """
    _, analysis = estimate_text_column(df)

    if "error" in analysis:
        return []

    all_scores = analysis.get("all_scores", {})

    # スコア順にソート
    sorted_columns = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"column": col, "score": score, "details": analysis["analysis_details"][col]}
        for col, score in sorted_columns[:top_n]
    ]
