import streamlit as st
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic # Using Anthropic as the LLM orchestrator

st.set_page_config(page_title="MCP Multi-Server Hub")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.title("Settings")
    anthropic_key = st.text_input("Anthropic API Key", type="password")
    github_token = st.text_input("GitHub Token (for GitHub MCP)", type="password")
    # For Context7, we use the library-neutral prompt prefix method
    context7_key = st.text_input("Context7 API Key", type="password")

# --- CORE MCP LOGIC ---
async def query_mcp_hub(user_prompt):
    # Define the GitHub MCP Server (runs via npx)
    github_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token}
    )

    async with stdio_client(github_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Fetch tools available from GitHub MCP
            tools = await session.list_tools()
            
            # Orchestrate with LLM
            client = anthropic.Anthropic(api_key=anthropic_key)
            
            # Inject Context7 "Global Mode" into the prompt
            final_prompt = f"use context7. {user_prompt}"
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": final_prompt}]
                # In a full app, you would pass 'tools' here for the LLM to call
            )
            return response.content[0].text

# --- UI INTERFACE ---
st.title("🤖 MCP Multi-Repo Assistant")
st.caption("Combining Context7 documentation with your specific MCP servers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a cross-repo question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not (anthropic_key and github_token):
        st.error("Please provide your API keys in the sidebar.")
    else:
        with st.spinner("Consulting MCP Servers..."):
            answer = asyncio.run(query_mcp_hub(prompt))
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.chat_message("assistant").write(answer)
