import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client  # ✅ Correct module in MCP ≥1.2

# --- 1. Configuration & Secrets ---
try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("Missing 'ANTHROPIC_API_KEY' in Streamlit Secrets!")
    st.stop()

MCP_CONFIG = {
    "hdsdocs": "https://gitmcp.io",
    "standard_pipeline": "https://gitmcp.io",
    "dbplyrdocs": "https://gitmcp.io"
}

st.set_page_config(page_title="Multi-Repo MCP Hub", page_icon="🔗")

# --- 2. Tool-Calling Loop ---
async def run_conversation(user_query):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    all_tools = []

    for name, url in MCP_CONFIG.items():
        try:
            # ✅ streamablehttp_client returns (read, write, get_session_id) — 3-tuple
            async with streamablehttp_client(url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    for t in tools:
                        t.name = f"{name}_{t.name}"
                        all_tools.append(t)
        except Exception as e:
            st.sidebar.warning(f"Could not reach {name}: {str(e)}")

    final_prompt = f"use context7. {user_query}"
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": final_prompt}]
    )
    return response.content[0].text  # ✅ content is a list, not a string

# --- 3. UI Interface ---
st.title("🌐 Multi-Repo MCP Assistant")
st.info(f"Connected via HTTP to: {', '.join(MCP_CONFIG.keys())}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your pipelines or any public library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching docs via HTTP..."):
            answer = asyncio.run(run_conversation(prompt))
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
