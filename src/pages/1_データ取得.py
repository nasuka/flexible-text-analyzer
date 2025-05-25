import re
import os

import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def extract_video_id(url):
    """YouTube URLからIDを抽出"""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:embed\/)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


class YouTubeCommentFetcher:
    def __init__(self, api_key):
        self.youtube = build("youtube", "v3", developerKey=api_key)
        self.comments = []
        self.comment_no = 1

    def fetch_comments(self, video_id, max_results=1000):
        """動画のコメントを取得"""
        try:
            self._fetch_top_level_comments(video_id, None)
            return self.comments
        except HttpError as e:
            st.error(f"YouTube API エラー: {e}")
            return []
        except Exception as e:
            st.error(f"エラー: {e}")
            return []

    def _fetch_top_level_comments(self, video_id, next_page_token):
        """トップレベルのコメントを取得"""
        try:
            request = self.youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order="relevance",
                pageToken=next_page_token,
            )
            response = request.execute()

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                reply_count = item["snippet"]["totalReplyCount"]
                comment_id = item["snippet"]["topLevelComment"]["id"]

                # コメントを追加
                self.comments.append(
                    {
                        "no": f"{self.comment_no:04d}",
                        "comment_id": comment_id,
                        "author": comment["authorDisplayName"],
                        "author_channel_id": comment.get("authorChannelId", {}).get(
                            "value", ""
                        ),
                        "author_channel_url": comment.get("authorChannelUrl", ""),
                        "comment": comment["textDisplay"]
                        .replace("\r", "\n")
                        .replace("\n", " "),
                        "likes": comment["likeCount"],
                        "published_at": comment["publishedAt"],
                        "updated_at": comment["updatedAt"],
                        "reply_count": reply_count,
                        "parent_no": "",
                        "reply_no": "",
                        "is_reply": False,
                        "moderation_status": comment.get("moderationStatus", ""),
                        "viewer_rating": comment.get("viewerRating", ""),
                        "can_rate": comment.get("canRate", False),
                        "can_reply": comment.get("canReply", False),
                        "is_public": comment.get("isPublic", True),
                    }
                )

                # 返信がある場合は取得
                if reply_count > 0:
                    self._fetch_replies(comment_id, self.comment_no)

                self.comment_no += 1

            # 次のページがある場合は再帰的に取得
            if "nextPageToken" in response:
                self._fetch_top_level_comments(video_id, response["nextPageToken"])

        except HttpError as e:
            if e.resp.status == 403:
                st.warning(
                    "APIのクォータ制限に達しました。一部のコメントのみ取得できました。"
                )
            else:
                raise e

    def _fetch_replies(self, parent_id, parent_no, next_page_token=None):
        """コメントへの返信を取得"""
        try:
            request = self.youtube.comments().list(
                part="snippet",
                parentId=parent_id,
                maxResults=50,
                pageToken=next_page_token,
            )
            response = request.execute()

            for i, item in enumerate(response["items"], 1):
                comment = item["snippet"]
                self.comments.append(
                    {
                        "no": f"{parent_no:04d}-{i:03d}",
                        "comment_id": item["id"],
                        "author": comment["authorDisplayName"],
                        "author_channel_id": comment.get("authorChannelId", {}).get(
                            "value", ""
                        ),
                        "author_channel_url": comment.get("authorChannelUrl", ""),
                        "comment": comment["textDisplay"]
                        .replace("\r", "\n")
                        .replace("\n", " "),
                        "likes": comment["likeCount"],
                        "published_at": comment["publishedAt"],
                        "updated_at": comment["updatedAt"],
                        "reply_count": 0,
                        "parent_no": f"{parent_no:04d}",
                        "reply_no": f"{i:03d}",
                        "is_reply": True,
                        "moderation_status": comment.get("moderationStatus", ""),
                        "viewer_rating": comment.get("viewerRating", ""),
                        "can_rate": comment.get("canRate", False),
                        "can_reply": comment.get("canReply", False),
                        "is_public": comment.get("isPublic", True),
                    }
                )

            # 次のページがある場合は再帰的に取得
            if "nextPageToken" in response:
                self._fetch_replies(parent_id, parent_no, response["nextPageToken"])

        except HttpError as e:
            if e.resp.status == 403:
                st.warning(
                    "APIのクォータ制限に達しました。一部の返信のみ取得できました。"
                )
            else:
                raise e


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
