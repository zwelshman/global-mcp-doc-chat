import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
import re

# — 1. Secrets —

try:
ANTHROPIC_KEY = st.secrets[“ANTHROPIC_API_KEY”]
except KeyError:
st.error(“Missing ‘ANTHROPIC_API_KEY’ in Streamlit Secrets!”)
st.stop()

GITMCP_BASE = “https://gitmcp.io/”

def is_valid_gitmcp_url(url: str) -> bool:
“”“Only allow gitmcp.io URLs pointing to valid GitHub owner/repo paths.”””
pattern = r”^https://gitmcp.io/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$”
return bool(re.match(pattern, url.strip()))

def normalise_gitmcp_url(raw: str) -> str:
“”“Accept ‘owner/repo’ shorthand or a full gitmcp.io URL.”””
raw = raw.strip().rstrip(”/”)
if raw.startswith(“https://gitmcp.io/”):
return raw
if “/” in raw and not raw.startswith(“http”):
return f”{GITMCP_BASE}{raw}”
return raw

# — 2. Default MCP config (shown on first load) —

DEFAULT_MCP_SERVERS = [
{“shortname”: “hdsdocs”,    “url”: “https://gitmcp.io/BHFDSC/documentation”},
{“shortname”: “dbplyrdocs”, “url”: “https://gitmcp.io/tidyverse/dbplyr”},
]

EXAMPLE_SHORTNAME = “pysparkdocs”
EXAMPLE_URL       = “https://gitmcp.io/apache/spark”
EXAMPLE_QUERY     = (
“Using the PySpark docs, generate a query to group by disease “
“and sum the number of persons.”
)

# — 3. Session state init —

if “mcp_servers” not in st.session_state:
st.session_state.mcp_servers = [dict(s) for s in DEFAULT_MCP_SERVERS]
if “messages” not in st.session_state:
st.session_state.messages = []

# — 4. Agentic loop —

async def run_conversation(user_query: str, mcp_config: dict, status) -> str:
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
tool_registry: dict[str, tuple[ClientSession, str]] = {}
anthropic_tools = []

```
async with AsyncExitStack() as stack:
    for name, url in mcp_config.items():
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

            if text_parts:
                status.write(f"💭 *{' '.join(text_parts)}*")

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

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
```

# — 5. UI —

st.set_page_config(page_title=“GitMCP Doc Assistant”, page_icon=“🔗”, layout=“wide”)

st.title(“🔗 GitMCP Doc Assistant”)
st.caption(
“Ask questions answered from GitHub-hosted documentation via “
“[gitmcp.io](https://gitmcp.io). **Only gitmcp.io servers are supported.**”
)

# ── Sidebar: MCP server configuration ──────────────────────────────────────

with st.sidebar:
st.header(“⚙️ MCP Server Config”)

```
st.markdown(
    "Add the GitHub repos whose docs you want to query. "
    "Each entry maps a **short name** (used as a tool prefix) to a "
    "`gitmcp.io` URL.\n\n"
    "You can paste either a full URL or `owner/repo` shorthand."
)

# ── Example callout ────────────────────────────────────────────────────
with st.expander("💡 Example", expanded=True):
    st.markdown(
        f"""
```

**Short name:** `{EXAMPLE_SHORTNAME}`  
**GitMCP URL:** `{EXAMPLE_URL}`

**Try asking:**

> *{EXAMPLE_QUERY}*
> “””
> )

```
st.divider()

# ── Editable server list ───────────────────────────────────────────────
st.subheader("Your servers")

servers_to_keep = []
for i, server in enumerate(st.session_state.mcp_servers):
    col_name, col_url, col_del = st.columns([2, 4, 1])
    with col_name:
        new_name = st.text_input(
            "Short name",
            value=server["shortname"],
            key=f"name_{i}",
            label_visibility="collapsed",
            placeholder="e.g. pysparkdocs",
        )
    with col_url:
        new_url = st.text_input(
            "GitMCP URL",
            value=server["url"],
            key=f"url_{i}",
            label_visibility="collapsed",
            placeholder="gitmcp.io/owner/repo  or  owner/repo",
        )
    with col_del:
        remove = st.button("✕", key=f"del_{i}", help="Remove this server")

    if not remove:
        servers_to_keep.append({"shortname": new_name, "url": new_url})

st.session_state.mcp_servers = servers_to_keep

# ── Add new row ────────────────────────────────────────────────────────
if st.button("➕ Add server"):
    st.session_state.mcp_servers.append({"shortname": "", "url": ""})
    st.rerun()

st.divider()

# ── Validate & build live config ───────────────────────────────────────
errors = []
mcp_config: dict[str, str] = {}
seen_names: set[str] = set()

for s in st.session_state.mcp_servers:
    name = s["shortname"].strip()
    raw_url = s["url"].strip()

    if not name and not raw_url:
        continue  # skip blank rows silently

    if not name:
        errors.append("⚠️ A server entry is missing a short name.")
        continue

    if not re.match(r"^[A-Za-z0-9_]+$", name):
        errors.append(f"⚠️ `{name}`: short names may only contain letters, digits, or underscores.")
        continue

    if name in seen_names:
        errors.append(f"⚠️ Duplicate short name: `{name}`.")
        continue

    resolved = normalise_gitmcp_url(raw_url)
    if not is_valid_gitmcp_url(resolved):
        errors.append(
            f"⚠️ `{name}`: URL must be a gitmcp.io path "
            f"(`https://gitmcp.io/owner/repo`). Got: `{raw_url}`"
        )
        continue

    seen_names.add(name)
    mcp_config[name] = resolved

if errors:
    for e in errors:
        st.warning(e)

if mcp_config:
    st.success(f"✅ {len(mcp_config)} server(s) configured")
    for name, url in mcp_config.items():
        st.markdown(f"- **{name}** → `{url}`")
else:
    st.info("No valid servers configured yet.")

st.divider()
if st.button("🗑️ Clear chat history"):
    st.session_state.messages = []
    st.rerun()
```

# ── Main chat area ──────────────────────────────────────────────────────────

if not mcp_config:
st.warning(
“⬅️ Configure at least one valid GitMCP server in the sidebar to start chatting.”
)
st.stop()

# Show active servers as pills

active_pills = “  “.join(
f”`{name}` → <{url}>” for name, url in mcp_config.items()
)
st.info(f”**Active servers:** {active_pills}”)

# Chat history

for msg in st.session_state.messages:
with st.chat_message(msg[“role”]):
st.markdown(msg[“content”])

# Chat input — hint uses current server names

server_names = “ · “.join(mcp_config.keys())
placeholder = f”Ask about docs from: {server_names}…”

if prompt := st.chat_input(placeholder):
st.session_state.messages.append({“role”: “user”, “content”: prompt})
with st.chat_message(“user”):
st.markdown(prompt)

```
with st.chat_message("assistant"):
    with st.status("🚀 Starting...", expanded=True) as status:
        answer = asyncio.run(run_conversation(prompt, mcp_config, status))
    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
```
