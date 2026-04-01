import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.http import http_client # Changed from sse_client

# --- 1. Configuration & Secrets ---
try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("Missing 'ANTHROPIC_API_KEY' in Streamlit Secrets!")
    st.stop()

# Updated URLs to the correct MCP HTTP endpoints (often /mcp or /)
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
    
    # 1. Connect to all HTTP-based MCP servers
    for name, url in MCP_CONFIG.items():
        try:
            # Use http_client for Streamable HTTP transport
            async with http_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    for t in tools:
                        # Prefixing tool names ensures they are unique
                        t.name = f"{name}_{t.name}"
                        all_tools.append(t)
        except Exception as e:
            st.sidebar.warning(f"Could not reach {name}: {str(e)}")

    # 2. Call the AI with Context7 global search instruction
    final_prompt = f"use context7. {user_query}"
    
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": final_prompt}]
        # tools=all_tools # Enable this for full tool execution
    )
    return response.content.text

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
