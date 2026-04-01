import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack

# --- 1. Secrets ---
try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("Missing 'ANTHROPIC_API_KEY' in Streamlit Secrets!")
    st.stop()

MCP_CONFIG = {
    "hdsdocs":           "https://gitmcp.io/BHFDSC/documentation",
    "standard_pipeline": "https://gitmcp.io/BHFDSC/documentation",
    "dbplyrdocs":        "https://gitmcp.io/tidyverse/dbplyr",
}

st.set_page_config(page_title="Multi-Repo MCP Hub", page_icon="🔗")


# --- 2. Agentic loop ---
async def run_conversation(user_query: str, status) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    tool_registry: dict[str, tuple[ClientSession, str]] = {}
    anthropic_tools = []

    async with AsyncExitStack() as stack:
        # Connect to MCP servers
        for name, url in MCP_CONFIG.items():
            status.update(label=f"🔌 Connecting to **{name}**...")
            try:
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(url)
                )
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()

                result = await session.list_tools()
                for t in result.tools:
                    prefixed = f"{name}_{t.name}"
                    tool_registry[prefixed] = (session, t.name)
                    anthropic_tools.append({
                        "name": prefixed,
                        "description": t.description or "",
                        "input_schema": t.inputSchema,
                    })
                status.write(f"✅ **{name}** — {len(result.tools)} tools loaded")
            except Exception as e:
                status.write(f"⚠️ **{name}** unreachable: `{e}`")

        if not tool_registry:
            return "❌ No MCP servers could be reached."

        tool_names = ", ".join(f"`{t}`" for t in tool_registry)
        status.write(f"🧰 Tools available: {tool_names}")

        # Agentic loop
        messages = [{"role": "user", "content": user_query}]
        turn = 0

        while True:
            turn += 1
            status.update(label=f"🤔 Thinking... (turn {turn})")

            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=4096,
                tools=anthropic_tools,
                messages=messages,
            )

            text_parts = [b.text for b in response.content if b.type == "text"]

            if response.stop_reason == "end_turn":
                status.update(label="✅ Done", state="complete", expanded=False)
                return "\n".join(text_parts) if text_parts else "(no response)"

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                # Show any reasoning text the model emitted before tool calls
                if text_parts:
                    status.write(f"💭 *{' '.join(text_parts)}*")

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue

                    # Show the tool call
                    status.update(label=f"🔧 Calling `{block.name}`...")
                    with status.container():
                        with st.expander(f"🔧 `{block.name}`", expanded=False):
                            st.json(block.input)

                    if block.name not in tool_registry:
                        result_text = f"Unknown tool: {block.name}"
                        status.write(f"❌ Unknown tool `{block.name}`")
                    else:
                        session, orig_name = tool_registry[block.name]
                        try:
                            call_result = await session.call_tool(orig_name, block.input)
                            result_text = "\n".join(
                                c.text for c in call_result.content if hasattr(c, "text")
                            ) or "(empty result)"
                            # Show a truncated preview of the result
                            preview = result_text[:300] + "..." if len(result_text) > 300 else result_text
                            status.write(f"↩️ `{block.name}` returned: {preview}")
                        except Exception as e:
                            result_text = f"Tool error: {e}"
                            status.write(f"❌ `{block.name}` error: `{e}`")

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results})

            else:
                status.update(label=f"⚠️ Unexpected stop: {response.stop_reason}", state="error")
                text_parts = [b.text for b in response.content if b.type == "text"]
                return "\n".join(text_parts) if text_parts else f"Stopped: {response.stop_reason}"


# --- 3. UI ---
st.title("🌐 Multi-Repo MCP Assistant")
st.info(f"MCP servers: {', '.join(MCP_CONFIG.keys())}")

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
        # st.status gives a live expandable activity log
        with st.status("🚀 Starting...", expanded=True) as status:
            answer = asyncio.run(run_conversation(prompt, status))
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
