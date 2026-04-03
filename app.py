import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
import re

# —————————————————————————

# 1. Secrets

# —————————————————————————

try:
ANTHROPIC_KEY = st.secrets[“ANTHROPIC_API_KEY”]
except KeyError:
st.error(“Missing ‘ANTHROPIC_API_KEY’ in Streamlit Secrets!”)
st.stop()

GITMCP_BASE = “https://gitmcp.io/”

# —————————————————————————

# 2. Library catalogue  (shortname, github_path, language, description)

# —————————————————————————

LIBRARY_CATALOGUE = [
# ── Python ──────────────────────────────────────────────────────────────
{“shortname”: “pandas”,      “github”: “pandas-dev/pandas”,          “lang”: “Python”, “desc”: “Data structures & analysis”,       “default”: True},
{“shortname”: “numpy”,       “github”: “numpy/numpy”,                 “lang”: “Python”, “desc”: “Scientific computing”,             “default”: True},
{“shortname”: “sklearn”,     “github”: “scikit-learn/scikit-learn”,   “lang”: “Python”, “desc”: “Machine learning”,                 “default”: True},
{“shortname”: “matplotlib”,  “github”: “matplotlib/matplotlib”,       “lang”: “Python”, “desc”: “Plotting & visualisation”,         “default”: False},
{“shortname”: “seaborn”,     “github”: “mwaskom/seaborn”,             “lang”: “Python”, “desc”: “Statistical visualisation”,        “default”: False},
{“shortname”: “scipy”,       “github”: “scipy/scipy”,                 “lang”: “Python”, “desc”: “Scientific & engineering maths”,   “default”: False},
{“shortname”: “statsmodels”, “github”: “statsmodels/statsmodels”,     “lang”: “Python”, “desc”: “Statistical modelling & tests”,    “default”: False},
{“shortname”: “pyspark”,     “github”: “apache/spark”,                “lang”: “Python”, “desc”: “Distributed data processing”,      “default”: False},
{“shortname”: “polars”,      “github”: “pola-rs/polars”,              “lang”: “Python”, “desc”: “Fast DataFrames (Rust-backed)”,    “default”: False},
{“shortname”: “pytorch”,     “github”: “pytorch/pytorch”,             “lang”: “Python”, “desc”: “Deep learning framework”,          “default”: False},
{“shortname”: “xgboost”,     “github”: “dmlc/xgboost”,               “lang”: “Python”, “desc”: “Gradient boosting”,                “default”: False},
{“shortname”: “lightgbm”,    “github”: “microsoft/LightGBM”,          “lang”: “Python”, “desc”: “Fast gradient boosting”,           “default”: False},
# ── R ───────────────────────────────────────────────────────────────────
{“shortname”: “dbplyr”,      “github”: “tidyverse/dbplyr”,            “lang”: “R”,      “desc”: “dplyr backend for databases”,      “default”: True},
{“shortname”: “dplyr”,       “github”: “tidyverse/dplyr”,             “lang”: “R”,      “desc”: “Data manipulation grammar”,        “default”: True},
{“shortname”: “ggplot2”,     “github”: “tidyverse/ggplot2”,           “lang”: “R”,      “desc”: “Grammar of graphics”,              “default”: False},
{“shortname”: “tidyr”,       “github”: “tidyverse/tidyr”,             “lang”: “R”,      “desc”: “Tidy data reshaping”,              “default”: False},
{“shortname”: “purrr”,       “github”: “tidyverse/purrr”,             “lang”: “R”,      “desc”: “Functional programming tools”,     “default”: False},
{“shortname”: “caret”,       “github”: “topepo/caret”,                “lang”: “R”,      “desc”: “ML training & tuning”,             “default”: False},
{“shortname”: “tidymodels”,  “github”: “tidymodels/tidymodels”,       “lang”: “R”,      “desc”: “Tidy modelling framework”,         “default”: False},
{“shortname”: “data.table”,  “github”: “Rdatatable/data.table”,       “lang”: “R”,      “desc”: “Fast in-memory data wrangling”,    “default”: False},
]

LANG_EMOJI = {“Python”: “🐍”, “R”: “📊”}

EXAMPLE_QUERY = (
“Using the pandas docs, generate code to group by a disease column “
“and sum the number of persons.”
)

def gitmcp_url(github: str) -> str:
return f”{GITMCP_BASE}{github}”

def is_valid_gitmcp_url(url: str) -> bool:
pattern = r”^https://gitmcp.io/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/?$”
return bool(re.match(pattern, url.strip()))

def normalise_gitmcp_url(raw: str) -> str:
raw = raw.strip().rstrip(”/”)
if raw.startswith(“https://gitmcp.io/”):
return raw
if “/” in raw and not raw.startswith(“http”):
return f”{GITMCP_BASE}{raw}”
return raw

# —————————————————————————

# 3. Session-state init

# —————————————————————————

if “active_shortnames” not in st.session_state:
st.session_state.active_shortnames = {
lib[“shortname”] for lib in LIBRARY_CATALOGUE if lib[“default”]
}
if “custom_servers” not in st.session_state:
st.session_state.custom_servers = []
if “messages” not in st.session_state:
st.session_state.messages = []

# —————————————————————————

# 4. Agentic loop

# —————————————————————————

async def run_conversation(user_query: str, mcp_config: dict, status) -> str:
client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
tool_registry: dict[str, tuple] = {}
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
            status.write(f"✅ **{name}** -- {len(result.tools)} tools loaded")
        except Exception as e:
            status.write(f"⚠️ **{name}** unreachable: `{e}`")

    if not tool_registry:
        return "❌ No MCP servers could be reached."

    status.write(
        "🧰 Tools available: "
        + ", ".join(f"`{t}`" for t in tool_registry)
    )

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
                        preview = (
                            result_text[:300] + "..."
                            if len(result_text) > 300
                            else result_text
                        )
                        status.write(f"↩️ `{block.name}`: {preview}")
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
            status.update(
                label=f"⚠️ Unexpected stop: {response.stop_reason}", state="error"
            )
            return (
                "\n".join(text_parts)
                if text_parts
                else f"Stopped: {response.stop_reason}"
            )
```

# —————————————————————————

# 5. UI

# —————————————————————————

st.set_page_config(
page_title=“GitMCP Doc Assistant”, page_icon=“🔗”, layout=“wide”
)

st.title(“🔗 GitMCP Doc Assistant”)
st.caption(
“Query GitHub-hosted documentation via [gitmcp.io](https://gitmcp.io). “
“**Only gitmcp.io servers are permitted.**”
)

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
st.header(“📚 Library Catalogue”)
st.markdown(
“Tick the libraries you want to query. “
“✅ = already active by default.”
)

```
for lang in ["Python", "R"]:
    libs = [l for l in LIBRARY_CATALOGUE if l["lang"] == lang]
    st.subheader(f"{LANG_EMOJI[lang]} {lang}")

    cols = st.columns(2)
    for idx, lib in enumerate(libs):
        col = cols[idx % 2]
        is_active = lib["shortname"] in st.session_state.active_shortnames
        # Tooltip shows description + URL
        help_text = (
            f"**{lib['desc']}**\n\n"
            f"`{gitmcp_url(lib['github'])}`"
        )
        checked = col.checkbox(
            lib["shortname"],
            value=is_active,
            key=f"cb_{lib['shortname']}",
            help=help_text,
        )
        if checked:
            st.session_state.active_shortnames.add(lib["shortname"])
        else:
            st.session_state.active_shortnames.discard(lib["shortname"])

st.divider()

# ── Custom servers ──────────────────────────────────────────────────────
st.subheader("🔧 Custom GitMCP server")
with st.expander("💡 Example", expanded=False):
    st.markdown(
        "**Short name:** `pysparkdocs`  \n"
        "**URL:** `apache/spark`  \n\n"
        f"*Try:* _{EXAMPLE_QUERY}_"
    )

custom_to_keep = []
for i, cs in enumerate(st.session_state.custom_servers):
    c1, c2, c3 = st.columns([2, 4, 1])
    new_name = c1.text_input(
        "Name", value=cs["shortname"], key=f"cname_{i}",
        label_visibility="collapsed", placeholder="shortname",
    )
    new_url = c2.text_input(
        "URL", value=cs["url"], key=f"curl_{i}",
        label_visibility="collapsed", placeholder="owner/repo",
    )
    remove = c3.button("✕", key=f"cdel_{i}")
    if not remove:
        custom_to_keep.append({"shortname": new_name, "url": new_url})

st.session_state.custom_servers = custom_to_keep

if st.button("➕ Add custom server"):
    st.session_state.custom_servers.append({"shortname": "", "url": ""})
    st.rerun()

st.divider()

# ── Validate & build final config ───────────────────────────────────────
errors = []
mcp_config: dict[str, str] = {}
seen_names: set[str] = set()

for lib in LIBRARY_CATALOGUE:
    if lib["shortname"] in st.session_state.active_shortnames:
        mcp_config[lib["shortname"]] = gitmcp_url(lib["github"])
        seen_names.add(lib["shortname"])

for cs in st.session_state.custom_servers:
    name = cs["shortname"].strip()
    raw = cs["url"].strip()
    if not name and not raw:
        continue
    if not name:
        errors.append("⚠️ Custom server missing a short name.")
        continue
    if not re.match(r"^[A-Za-z0-9_]+$", name):
        errors.append(f"⚠️ `{name}`: letters, digits, underscores only.")
        continue
    if name in seen_names:
        errors.append(f"⚠️ Duplicate short name: `{name}`.")
        continue
    resolved = normalise_gitmcp_url(raw)
    if not is_valid_gitmcp_url(resolved):
        errors.append(f"⚠️ `{name}`: must be a gitmcp.io/owner/repo URL.")
        continue
    seen_names.add(name)
    mcp_config[name] = resolved

for e in errors:
    st.warning(e)

if mcp_config:
    st.success(f"✅ **{len(mcp_config)} server(s) active**")
    for n, u in mcp_config.items():
        st.markdown(f"- **{n}** → `{u}`")
else:
    st.info("Select at least one library to start.")

st.divider()
if st.button("🗑️ Clear chat history"):
    st.session_state.messages = []
    st.rerun()
```

# ── Main chat area ──────────────────────────────────────────────────────────

if not mcp_config:
st.warning(“⬅️ Select at least one library from the sidebar to start chatting.”)
st.stop()

# Active-library status bar

catalogue_names = {l[“shortname”] for l in LIBRARY_CATALOGUE}
py_active = [
l[“shortname”] for l in LIBRARY_CATALOGUE
if l[“lang”] == “Python” and l[“shortname”] in mcp_config
]
r_active = [
l[“shortname”] for l in LIBRARY_CATALOGUE
if l[“lang”] == “R” and l[“shortname”] in mcp_config
]
custom_active = [n for n in mcp_config if n not in catalogue_names]

parts = []
if py_active:
parts.append(f”🐍 **Python:** {’, ‘.join(f’`{n}`’ for n in py_active)}”)
if r_active:
parts.append(f”📊 **R:** {’, ‘.join(f’`{n}`’ for n in r_active)}”)
if custom_active:
parts.append(f”🔧 **Custom:** {’, ‘.join(f’`{n}`’ for n in custom_active)}”)

st.info(”  \n”.join(parts))

# Chat history

for msg in st.session_state.messages:
with st.chat_message(msg[“role”]):
st.markdown(msg[“content”])

# Dynamic placeholder

shown = list(mcp_config.keys())[:4]
suffix = “…” if len(mcp_config) > 4 else “”
placeholder = “Ask about “ + “, “.join(shown) + suffix

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