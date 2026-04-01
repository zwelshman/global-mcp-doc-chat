import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client  # ✅ non-deprecated name

# --- 1. Configuration & Secrets ---
try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("Missing 'ANTHROPIC_API_KEY' in Streamlit Secrets!")
    st.stop()

# ✅ Full repo paths required: https://gitmcp.io/{owner}/{repo}
MCP_CONFIG = {
    "hdsdocs":           "https://gitmcp.io/BHFDSC/documentation",
    "standard_pipeline": "https://gitmcp.io/BHFDSC/documentation",   # replace with real repo
    "dbplyrdocs":        "https://gitmcp.io/tidyverse/dbplyr",        # replace with real repo
}

st.set_page_config(page_title="Multi-Repo MCP Hub", page_icon="🔗")

# --- 2. Tool-Calling Loop ---
async def connect_mcp(name: str, url: str) -> list:
    """Connect to one MCP server and return its tools, or [] on failure."""
    tools = []
    try:
        async with streamable_http_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                for t in result.tools:
                    t.name = f"{name}_{t.name}"
                    tools.append(t)
    except* Exception as eg:          # ✅ catches ExceptionGroup from TaskGroup
        for exc in eg.exceptions:
            st.sidebar.warning(f"Could not reach **{name}**: {type(exc).__name__}: {exc}")
    except Exception as e:            # plain exceptions (non-TaskGroup)
        st.sidebar.warning(f"Could not reach **{name}**: {type(e).__name__}: {e}")
    return tools


async def run_conversation(user_query: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

    # Gather tools from all MCP servers concurrently
    results = await asyncio.gather(
        *[connect_mcp(name, url) for name, url in MCP_CONFIG.items()],
        return_exceptions=False,
    )
    all_tools = [t for tool_list in results for t in tool_list]

    response = client.messages.create(
        model="claude-sonnet-4-5",     # ✅ valid model string in SDK 0.87+
        max_tokens=1024,
        messages=[{"role": "user", "content": user_query}],
    )
    return response.content[0].text


# --- 3. UI ---
st.title("🌐 Multi-Repo MCP Assistant")
st.info(f"Connecting to: {', '.join(MCP_CONFIG.keys())}")

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
        with st.spinner("Thinking..."):
            answer = asyncio.run(run_conversation(prompt))
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
