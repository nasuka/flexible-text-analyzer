import os

import pandas as pd
import streamlit as st

from services.youtube_fetcher import YouTubeCommentFetcher, extract_video_id


def main():
    st.title("YouTube コメント取得")
    st.markdown("---")
    st.header("YouTube コメント取得")
    with st.form("youtube_form"):
        col1, col2 = st.columns([3, 1])
        with col1:
            youtube_url = st.text_input(
                "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
            )
        with col2:
            max_comments = st.number_input(
                "取得件数", min_value=1, max_value=1000, value=1000
            )
        api_key = st.text_input(
            "YouTube API Key",
            value=os.getenv("YOUTUBE_API_KEY"),
            type="password",
            help="YouTube Data API v3のAPIキー",
        )
        submitted = st.form_submit_button("取得", type="primary")
    if submitted:
        if not youtube_url:
            st.error("YouTube URLが入力されていません")
        elif not api_key:
            st.error("APIキーが入力されていません")
        else:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("YouTube URLが不正です")
            else:
                with st.spinner("取得中..."):
                    fetcher = YouTubeCommentFetcher(api_key)
                    comments = fetcher.fetch_comments(video_id, max_comments)
                    if comments:
                        st.success(f"{len(comments)}件取得しました")
                        df = pd.DataFrame(comments)
                        st.subheader("統計情報")
                        st.metric("平均いいね数", f"{df['likes'].mean():.1f}")
                        st.metric("最大いいね数", df["likes"].max())
                        st.subheader("ソートオプション")
                        sort_option = st.selectbox(
                            "ソートオプション",
                            ["いいね数", "投稿日時", "投稿日時（新しい順）"],
                        )
                        if sort_option == "いいね数":
                            df_sorted = df.sort_values("likes", ascending=False)
                        elif sort_option == "投稿日時":
                            df_sorted = df.sort_values("published_at", ascending=False)
                        else:
                            df_sorted = df.sort_values("published_at", ascending=True)
                        st.dataframe(
                            df_sorted, use_container_width=True, hide_index=True
                        )
                        st.subheader("CSVダウンロード")
                        csv = df.to_csv(index=False, encoding="utf-8-sig")
                        st.download_button(
                            label="CSVダウンロード",
                            data=csv,
                            file_name=f"youtube_comments_{video_id}.csv",
                            mime="text/csv",
                        )
                        st.session_state["youtube_comments"] = df
                    else:
                        st.error("コメントの取得に失敗しました")
    if "youtube_comments" in st.session_state:
        st.markdown("---")
        st.info("コメントの取得に成功しました")
        st.info(f"取得件数: {len(st.session_state['youtube_comments'])}")


if __name__ == "__main__":
    main()
