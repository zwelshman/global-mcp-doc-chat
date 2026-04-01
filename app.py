import streamlit as st
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic

st.title("🤖 Cloud MCP Assistant")

# Use st.secrets for production security
# Setup these in the Streamlit Cloud Dashboard under 'Settings > Secrets'
try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    CONTEXT7_KEY = st.secrets["CONTEXT7_API_KEY"]
except KeyError:
    st.error("Missing Secrets! Add them in the Streamlit Cloud Dashboard.")
    st.stop()

async def run_mcp_query(user_prompt):
    # npx is now available because of packages.txt
    github_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": GITHUB_TOKEN}
    )

    async with stdio_client(github_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Prefix with 'use context7' for neutral library search
            client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": f"use context7. {user_prompt}"}]
            )
            return response.content[0].text

# --- Simple Chat UI ---
if prompt := st.chat_input("Ask anything..."):
    st.chat_message("user").write(prompt)
    with st.spinner("Connecting to MCP..."):
        answer = asyncio.run(run_mcp_query(prompt))
        st.chat_message("assistant").write(answer)
