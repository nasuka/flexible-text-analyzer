import streamlit as st

st.set_page_config(
    page_title="Mirai Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 Mirai Analyzer")
st.markdown("---")

st.markdown("""
## 📋 機能一覧

### 📊 データ取得
- **YouTube コメント取得**: YouTube動画のコメントを取得・分析
- **その他のデータソース**: 今後追加予定

### 🎯 クラス分類
- **テキスト分類**: 取得したテキストデータの分類・分析
- **感情分析**: コメントの感情分析

## 🚀 使い方
1. 左サイドバーから機能を選択
2. 必要な情報を入力
3. データを取得・分析

## 📝 注意事項
- YouTube APIキーが必要です
- APIの利用制限にご注意ください
""")

st.markdown("---")
st.markdown("**📍 左サイドバーから機能を選択してください**")
