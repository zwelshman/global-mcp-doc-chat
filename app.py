import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.sse import sse_client

# --- 1. Configuration & Secrets ---
# These must be set in the Streamlit Cloud Dashboard under Settings > Secrets
try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("Missing 'ANTHROPIC_API_KEY' in Streamlit Secrets!")
    st.stop()

# Your specific MCP HTTP/SSE endpoints
MCP_CONFIG = {
    "hdsdocs": "https://gitmcp.io",
    "standard_pipeline": "https://gitmcp.io",
    "dbplyrdocs": "https://gitmcp.io"
}

st.set_page_config(page_title="Multi-Repo MCP Hub", page_icon="🔗")

# --- 2. MCP Tool Fetcher ---
async def get_remote_mcp_tools():
    """Connects to all gitmcp.io servers and aggregates their tool definitions."""
    all_tools = []
    for name, url in MCP_CONFIG.items():
        try:
            # gitmcp.io uses SSE (Server-Sent Events) for transport
            async with sse_client(url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    # Add server metadata to tool names to prevent collisions
                    for tool in tools:
                        tool.name = f"{name}_{tool.name}"
                        all_tools.append(tool)
        except Exception as e:
            st.sidebar.warning(f"Could not reach {name}: {str(e)}")
    return all_tools

# --- 3. Chat Logic ---
async def run_conversation(user_query):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    
    # Prepend Context7 'library-neutral' instruction
    # This tells the model to also use Context7's global index
    global_prompt = f"use context7. {user_query}"
    
    # Fetch tools from your 3 specific gitmcp servers
    # Note: For full tool execution, you'd implement a loop here to handle tool_use
    remote_tools = await get_remote_mcp_tools()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[{"role": "user", "content": global_prompt}],
        tools=remote_tools # Requires a loop to execute the tool results
    )
    return response.content[0].text

# --- 4. Streamlit UI ---
st.title("🌐 Multi-Repo MCP Assistant")
st.info(f"Connected to: {', '.join(MCP_CONFIG.keys())} + Context7 Global")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle Input
if prompt := st.chat_input("Ask about your pipelines or any public library..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching docs..."):
            answer = asyncio.run(run_conversation(prompt))
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
