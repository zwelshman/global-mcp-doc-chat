import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.sse import sse_client

# 1. Configuration - Your Specific MCP Servers
MCP_CONFIG = {
    "hdsdocs": "https://gitmcp.io",
    "standard_pipeline": "https://gitmcp.io",
    "dbplyrdocs": "https://gitmcp.io"
}

st.set_page_config(page_title="Multi-Repo MCP Assistant", layout="wide")

# 2. Sidebar for API Keys (Or use st.secrets for Cloud Deployment)
with st.sidebar:
    st.title("Settings")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    st.info("On Streamlit Cloud, add 'ANTHROPIC_API_KEY' to your Secrets.")

# 3. Helper to connect to SSE Servers and fetch tools
async def get_tools_from_servers():
    all_tools = []
    for name, url in MCP_CONFIG.items():
        try:
            # Connect to remote gitmcp.io servers
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    # We tag tools with their origin for clarity
                    all_tools.extend(tools)
        except Exception as e:
            st.error(f"Error connecting to {name}: {e}")
    return all_tools

# 4. Main Chat Logic
async def handle_chat(user_query):
    client = anthropic.Anthropic(api_key=anthropic_key)
    
    # Prepend Context7 'Neutral' command
    enhanced_prompt = f"use context7. {user_query}"
    
    # Fetch tools from your 3 specific servers
    available_tools = await get_tools_from_servers()
    
    # Call Claude (Note: tool_use requires a loop in a full implementation)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": enhanced_prompt}]
        # tools=available_tools  # Uncomment when tool-handling loop is ready
    )
    return response.content[0].text

# --- UI INTERFACE ---
st.title("🔍 Multi-Repo Documentation Assistant")
st.markdown(f"**Connected to:** `{', '.join(MCP_CONFIG.keys())}` + **Context7 Global Search**")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("Ask about your pipelines or any public library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not anthropic_key:
        st.warning("Please enter your Anthropic API Key in the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Querying MCP servers and Context7..."):
                response_text = asyncio.run(handle_chat(prompt))
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
