import streamlit as st
import anthropic

try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("Missing 'ANTHROPIC_API_KEY' in Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Context7 Doc Chat", page_icon="📚")

def run_conversation(user_query: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    response = client.beta.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"use context7. {user_query}"}],
        mcp_servers=[
            {
                "type": "url",
                "url": "https://mcp.context7.com/mcp",
                "name": "context7",
            }
        ],
        betas=["mcp-client-2025-04-04"],
    )

    text_parts = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_parts) if text_parts else "(no response)"


st.title("📚 Context7 Doc Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about any library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Fetching docs..."):
            answer = run_conversation(prompt)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
