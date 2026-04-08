import streamlit as st
import asyncio
import anthropic
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
import re
import json
from datetime import datetime

# --- CONFIG & CONSTANTS ---
GITMCP_BASE = "https://gitmcp.io/"
MAX_ACTIVE_SERVERS = 2
INPUT_COST_PER_M  = 1.00
OUTPUT_COST_PER_M = 5.00

LIBRARY_CATALOGUE = [
    {"shortname": "bhf_docs", "github": "BHFDSC/documentation", "lang": "Python", "desc": "BHF DSC data curation documentation", "default": True},
    {"shortname": "pandas", "github": "pandas-dev/pandas", "lang": "Python", "desc": "Data structures & analysis", "default": False},
    {"shortname": "numpy", "github": "numpy/numpy", "lang": "Python", "desc": "Scientific computing", "default": False},
    {"shortname": "sklearn", "github": "scikit-learn/scikit-learn", "lang": "Python", "desc": "Machine learning", "default": False},
    {"shortname": "pyspark", "github": "apache/spark", "lang": "Python", "desc": "Distributed data processing", "default": False},
    {"shortname": "dplyr", "github": "tidyverse/dplyr", "lang": "R", "desc": "Data manipulation grammar", "default": False},
    {"shortname": "ggplot2", "github": "tidyverse/ggplot2", "lang": "R", "desc": "Grammar of graphics", "default": False},
]

# This defines which keys to look for in st.secrets
CUSTOM_MCP_CATALOGUE = [
    {"shortname": "context7", "secret_key": "context7", "desc": "Context7 library docs MCP"},
    {"shortname": "my_supabase", "secret_key": "my_supabase", "desc": "My Supabase MCP server"},
]

LANG_LABEL = {"Python": "Python", "R": "R"}

# --- HELPER FUNCTIONS ---

def unwrap_exc(e):
    if hasattr(e, "exceptions") and e.exceptions:
        return "; ".join(unwrap_exc(sub) for sub in e.exceptions)
    return str(e)

def gitmcp_url(github):
    return GITMCP_BASE + github

def is_valid_gitmcp_url(url):
    pattern = r"^https://gitmcp\.io/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/?$"
    return bool(re.match(pattern, url.strip()))

def normalise_gitmcp_url(raw):
    raw = raw.strip().rstrip("/")
    if raw.startswith("https://gitmcp.io/"): return raw
    if "/" in raw and not raw.startswith("http"): return GITMCP_BASE + raw
    return raw

def calc_cost(input_tokens, output_tokens):
    return (input_tokens / 1_000_000 * INPUT_COST_PER_M + output_tokens / 1_000_000 * OUTPUT_COST_PER_M)

def load_custom_mcp_from_secrets():
    available = {}
    for entry in CUSTOM_MCP_CATALOGUE:
        key = entry["secret_key"]
        try:
            if key in st.secrets:
                secret = st.secrets[key]
                url = secret.get("url", "").strip()
                if url:
                    available[entry["shortname"]] = {
                        "url": url,
                        "token": secret.get("token", "").strip(),
                        "desc": entry["desc"],
                    }
        except Exception:
            pass
    return available

def extract_version_from_description(description):
    if not description: return None
    patterns = [r"version[:\s]+([0-9]+\.[0-9]+[\.[0-9a-zA-Z]*]*)", r"v([0-9]+\.[0-9]+[\.[0-9a-zA-Z]*]*)", r"([0-9]+\.[0-9]+\.[0-9]+)"]
    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m: return m.group(1)
    return None

# --- ASYNC LOGIC ---

async def probe_servers(mcp_config, mcp_headers=None):
    if not mcp_headers: mcp_headers = {}
    results = {}
    async with AsyncExitStack() as stack:
        for name, url in mcp_config.items():
            try:
                headers = mcp_headers.get(name)
                http_client = await stack.enter_async_context(httpx.AsyncClient(headers=headers)) if headers else None
                read, write, _ = await stack.enter_async_context(streamable_http_client(url, http_client=http_client))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tool_list = await session.list_tools()
                version = next((extract_version_from_description(t.description or "") for t in tool_list.tools if extract_version_from_description(t.description or "")), None)
                results[name] = {"ok": True, "tools": len(tool_list.tools), "version": version, "error": None}
            except Exception as e:
                results[name] = {"ok": False, "tools": 0, "version": None, "error": unwrap_exc(e)}
    return results

async def run_conversation(user_query, mcp_config, status, mcp_headers=None):
    if not mcp_headers: mcp_headers = {}
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    tool_registry, anthropic_tools = {}, []
    input_tokens, output_tokens = 0, 0

    async with AsyncExitStack() as stack:
        for name, url in mcp_config.items():
            try:
                headers = mcp_headers.get(name)
                http_client = await stack.enter_async_context(httpx.AsyncClient(headers=headers)) if headers else None
                read, write, _ = await stack.enter_async_context(streamable_http_client(url, http_client=http_client))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                result = await session.list_tools()
                for t in result.tools:
                    prefixed = f"{name}_{t.name}"
                    tool_registry[prefixed] = (session, t.name)
                    anthropic_tools.append({"name": prefixed, "description": t.description or "", "input_schema": t.inputSchema})
            except Exception as e:
                status.write(f"⚠️ {name} failed: {unwrap_exc(e)}")

        if not tool_registry: return "No tools available.", 0, 0

        messages = [{"role": "user", "content": user_query}]
        while True:
            response = client.messages.create(model="claude-3-5-haiku-latest", max_tokens=4096, tools=anthropic_tools, messages=messages)
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            
            if response.stop_reason == "end_turn":
                return "".join([b.text for b in response.content if b.type == "text"]), input_tokens, output_tokens

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        session, orig_name = tool_registry[block.name]
                        try:
                            res = await session.call_tool(orig_name, block.input)
                            txt = "\n".join(c.text for c in res.content if hasattr(c, "text"))
                            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": txt})
                        except Exception as e:
                            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": f"Error: {str(e)}", "is_error": True})
                messages.append({"role": "user", "content": tool_results})

# --- UI SETUP ---
st.set_page_config(page_title="MCP Doc Assistant", layout="wide")

# Session States
for key, val in [("active_shortnames", {l["shortname"] for l in LIBRARY_CATALOGUE if l["default"]}), ("custom_servers", []), ("custom_mcp_servers", []), ("messages", []), ("connection_status", {}), ("total_input_tokens", 0), ("total_output_tokens", 0)]:
    if key not in st.session_state: st.session_state[key] = val

# --- SIDEBAR ---
with st.sidebar:
    st.header("MCP Settings")
    mcp_config, mcp_headers, seen_names, errors = {}, {}, set(), []
    
    # Global limit check
    available_custom = load_custom_mcp_from_secrets()
    total_active = len(st.session_state.active_shortnames)
    at_limit = total_active >= MAX_ACTIVE_SERVERS

    # 1. Library Catalogue
    st.subheader("Library Catalogue")
    for lang in ["Python", "R"]:
        st.markdown(f"**{lang}**")
        cols = st.columns(2)
        libs = [l for l in LIBRARY_CATALOGUE if l["lang"] == lang]
        for idx, lib in enumerate(libs):
            sname = lib["shortname"]
            is_active = sname in st.session_state.active_shortnames
            if col := cols[idx % 2].checkbox(sname, value=is_active, disabled=(at_limit and not is_active), key=f"lib_{sname}"):
                st.session_state.active_shortnames.add(sname)
                mcp_config[sname] = gitmcp_url(lib["github"])
                seen_names.add(sname)
            else:
                st.session_state.active_shortnames.discard(sname)

    # 2. Custom Secrets Catalogue
    if available_custom:
        st.divider()
        st.subheader("Configured Servers")
        cols = st.columns(2)
        for idx, (sname, meta) in enumerate(available_custom.items()):
            is_active = sname in st.session_state.active_shortnames
            if cols[idx % 2].checkbox(sname, value=is_active, disabled=(at_limit and not is_active), key=f"sec_{sname}", help=meta['desc']):
                st.session_state.active_shortnames.add(sname)
                mcp_config[sname] = meta["url"]
                seen_names.add(sname)
                if meta["token"]: mcp_headers[sname] = {"Authorization": f"Bearer {meta['token']}"}
            else:
                st.session_state.active_shortnames.discard(sname)

    # 3. Freeform Custom
    st.divider()
    st.subheader("Add Custom (Manual)")
    # (Simplified for brevity, standard manual entry logic goes here)

    # Sidebar Stats
    st.divider()
    cost = calc_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens)
    st.metric("Session Cost", f"${cost:.4f}")
    if st.button("Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT ---
st.title("MCP Doc Assistant")

if not mcp_config:
    st.info("Select a library in the sidebar to begin.")
    st.stop()

# Connection Probe
config_key = str(sorted(mcp_config.items()))
if st.session_state.connection_status.get("_key") != config_key:
    with st.status("Probing servers...") as s:
        st.session_state.connection_status = asyncio.run(probe_servers(mcp_config, mcp_headers))
        st.session_state.connection_status["_key"] = config_key
        s.update(label="Ready", state="complete")

# Display Status Table
with st.expander("Server Status", expanded=False):
    for name, url in mcp_config.items():
        stat = st.session_state.connection_status.get(name, {})
        icon = "✅" if stat.get("ok") else "❌"
        st.write(f"{icon} **{name}**: {stat.get('tools', 0)} tools | v{stat.get('version') or '?'}")

# Chat Interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the docs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)
    
    with st.chat_message("assistant"):
        status_placeholder = st.status("Thinking...")
        response_text, in_t, out_t = asyncio.run(run_conversation(prompt, mcp_config, status_placeholder, mcp_headers))
        st.markdown(response_text)
        st.session_state.total_input_tokens += in_t
        st.session_state.total_output_tokens += out_t
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        status_placeholder.update(label="Done", state="complete")
