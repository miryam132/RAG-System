"""app.py: ממשק Streamlit. כל הלוגיקה נמצאת ב-rag.py."""
import streamlit as st

import config
import rag

ALL_GAMES = "כל המשחקים"

st.set_page_config(page_title="עוזר חוקי משחקים", page_icon="🎲", layout="centered")
st.markdown(
    "<style>.stApp {direction: rtl; text-align: right;}</style>",
    unsafe_allow_html=True,
)
st.title("🎲 עוזר חוקי משחקי קופסה")
st.caption("שאל שאלה על חוקי המשחקים וקבל תשובה מבוססת מקורות.")


@st.cache_resource
def init_models():
    config.setup_embeddings()
    config.setup_llm()


@st.cache_resource
def get_index(label):
    return rag.load_index(label)


try:
    init_models()
    labels = rag.available_labels()
    if not labels:
        raise RuntimeError("לא נמצא מסד נתונים. הרץ קודם: python src/ingest.py")
except Exception as error:
    st.error(f"שגיאה באתחול המערכת: {error}")
    st.stop()

with st.sidebar:
    st.subheader("הגדרות")
    label = st.selectbox("הגדרת chunking", labels)
    game = st.selectbox("חיפוש במשחק", [ALL_GAMES] + rag.list_games(label))
    if st.button("נקה שיחה", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

index = get_index(label)
selected_game = None if game == ALL_GAMES else game

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(sources):
    with st.expander("מקורות"):
        for s in sources:
            st.markdown(f"**{s['game']} | {s['section']}** (ציון: {s['score']:.2f})")
            st.caption(s["text"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            show_sources(message["sources"])

question = st.chat_input("שאל שאלה על חוקי המשחקים...")
if question:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("מחפש בחוקים..."):
            try:
                result = rag.ask(index, question, selected_game, history)
            except Exception as error:
                st.error(f"אירעה שגיאה: {error}")
                st.session_state.messages.pop()
                st.stop()
        st.markdown(result["answer"])
        show_sources(result["sources"])
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )