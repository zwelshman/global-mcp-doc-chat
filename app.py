import streamlit as st
import asyncio
import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
import re

try:
    ANTHROPIC_KEY = st.secrets["ANTHROPIC_API_KEY"]
except KeyError:
    st.error("Missing ANTHROPIC_API_KEY in Streamlit Secrets!")
    st.stop()

GITMCP_BASE = "https://gitmcp.io/"
MAX_ACTIVE_SERVERS = 3

INPUT_COST_PER_M  = 1.00
OUTPUT_COST_PER_M = 5.00

LIBRARY_CATALOGUE = [
    {"shortname": "bhf_docs",                   "github": "BHFDSC/documentation",          "lang": "Python", "desc": "BHF DSC data curation documentation",         "default": True},
    {"shortname": "bhf_pyspark_standard_pipeline", "github": "BHFDSC/standard-pipeline",   "lang": "Python", "desc": "BHF DSC pyspark data curation pipeline",       "default": False},
    {"shortname": "bhf_ckd_phenotype",          "github": "BHFDSC/hds_phenotypes_ckd",    "lang": "Python", "desc": "BHF HDS DSC chronic kidney disease phenotype",  "default": False},
    {"shortname": "pandas",                     "github": "pandas-dev/pandas",             "lang": "Python", "desc": "Data structures & analysis",                   "default": False},
    {"shortname": "numpy",                      "github": "numpy/numpy",                   "lang": "Python", "desc": "Scientific computing",                         "default": False},
    {"shortname": "sklearn",                    "github": "scikit-learn/scikit-learn",     "lang": "Python", "desc": "Machine learning",                             "default": False},
    {"shortname": "matplotlib",                 "github": "matplotlib/matplotlib",         "lang": "Python", "desc": "Plotting & visualisation",                     "default": False},
    {"shortname": "seaborn",                    "github": "mwaskom/seaborn",               "lang": "Python", "desc": "Statistical visualisation",                    "default": False},
    {"shortname": "scipy",                      "github": "scipy/scipy",                   "lang": "Python", "desc": "Scientific & engineering maths",               "default": False},
    {"shortname": "statsmodels",                "github": "statsmodels/statsmodels",       "lang": "Python", "desc": "Statistical modelling & tests",                "default": False},
    {"shortname": "pyspark",                    "github": "apache/spark",                  "lang": "Python", "desc": "Distributed data processing",                  "default": False},
    {"shortname": "polars",                     "github": "pola-rs/polars",                "lang": "Python", "desc": "Fast DataFrames (Rust-backed)",               "default": False},
    {"shortname": "pytorch",                    "github": "pytorch/pytorch",               "lang": "Python", "desc": "Deep learning framework",                      "default": False},
    {"shortname": "xgboost",                    "github": "dmlc/xgboost",                 "lang": "Python", "desc": "Gradient boosting",                            "default": False},
    {"shortname": "lightgbm",                   "github": "microsoft/LightGBM",            "lang": "Python", "desc": "Fast gradient boosting",                       "default": False},
    {"shortname": "dbplyr",                     "github": "tidyverse/dbplyr",              "lang": "R",      "desc": "dplyr backend for databases",                  "default": False},
    {"shortname": "dplyr",                      "github": "tidyverse/dplyr",               "lang": "R",      "desc": "Data manipulation grammar",                    "default": False},
    {"shortname": "ggplot2",                    "github": "tidyverse/ggplot2",             "lang": "R",      "desc": "Grammar of graphics",                          "default": False},
    {"shortname": "tidyr",                      "github": "tidyverse/tidyr",               "lang": "R",      "desc": "Tidy data reshaping",                          "default": False},
    {"shortname": "purrr",                      "github": "tidyverse/purrr",               "lang": "R",      "desc": "Functional programming tools",                 "default": False},
    {"shortname": "caret",                      "github": "topepo/caret",                  "lang": "R",      "desc": "ML training & tuning",                         "default": False},
    {"shortname": "tidymodels",                 "github": "tidymodels/tidymodels",         "lang": "R",      "desc": "Tidy modelling framework",                     "default": False},
    {"shortname": "data.table",                 "github": "Rdatatable/data.table",         "lang": "R",      "desc": "Fast in-memory data wrangling",               "default": False},
]

LANG_LABEL = {"Python": "Python", "R": "R"}

EXAMPLE_QUERY = (
    "Using the pandas docs, generate code to group by a disease column "
    "and sum the number of persons."
)


def gitmcp_url(github):
    return GITMCP_BASE + github


def is_valid_gitmcp_url(url):
    pattern = r"^https://gitmcp\.io/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+/?$"
    return bool(re.match(pattern, url.strip()))


def normalise_gitmcp_url(raw):
    raw = raw.strip().rstrip("/")
    if raw.startswith("https://gitmcp.io/"):
        return raw
    if "/" in raw and not raw.startswith("http"):
        return GITMCP_BASE + raw
    return raw


def calc_cost(input_tokens, output_tokens):
    return (input_tokens / 1_000_000 * INPUT_COST_PER_M
            + output_tokens / 1_000_000 * OUTPUT_COST_PER_M)


def extract_version_from_description(description):
    if not description:
        return None
    patterns = [
        r"version[:\s]+([0-9]+\.[0-9]+[\.[0-9a-zA-Z]*]*)",
        r"v([0-9]+\.[0-9]+[\.[0-9a-zA-Z]*]*)",
        r"([0-9]+\.[0-9]+\.[0-9]+)",
    ]
    for pat in patterns:
        m = re.search(pat, description, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


if "active_shortnames" not in st.session_state:
    st.session_state.active_shortnames = {
        lib["shortname"] for lib in LIBRARY_CATALOGUE if lib["default"]
    }
if "custom_servers" not in st.session_state:
    st.session_state.custom_servers = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "connection_status" not in st.session_state:
    st.session_state.connection_status = {}
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0


async def probe_servers(mcp_config):
    results = {}
    async with AsyncExitStack() as stack:
        for name, url in mcp_config.items():
            try:
                read, write, _ = await stack.enter_async_context(streamable_http_client(url))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tool_list = await session.list_tools()
                tools = tool_list.tools
                version = None
                for t in tools:
                    version = extract_version_from_description(t.description or "")
                    if version:
                        break
                results[name] = {"ok": True, "tools": len(tools), "version": version, "error": None}
            except Exception as e:
                results[name] = {"ok": False, "tools": 0, "version": None, "error": str(e)}
    return results


async def run_conversation(user_query, mcp_config, status):
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    tool_registry = {}
    anthropic_tools = []
    input_tokens  = 0
    output_tokens = 0

    async with AsyncExitStack() as stack:
        for name, url in mcp_config.items():
            status.update(label="Connecting to " + name + "...")
            try:
                read, write, _ = await stack.enter_async_context(streamable_http_client(url))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                result = await session.list_tools()
                for t in result.tools:
                    prefixed = name + "_" + t.name
                    tool_registry[prefixed] = (session, t.name)
                    anthropic_tools.append({
                        "name": prefixed,
                        "description": t.description or "",
                        "input_schema": t.inputSchema,
                    })
                status.write("Connected: " + name + " (" + str(len(result.tools)) + " tools)")
            except Exception as e:
                status.write("Unreachable: " + name + " - " + str(e))

        if not tool_registry:
            return "No MCP servers could be reached.", 0, 0

        status.write("Tools available: " + ", ".join(tool_registry.keys()))
        messages = [{"role": "user", "content": user_query}]
        turn = 0

        while True:
            turn += 1
            status.update(label="Thinking... (turn " + str(turn) + ")")
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=4096,
                    tools=anthropic_tools,
                    messages=messages,
                )
            except anthropic.RateLimitError:
                status.update(label="Rate limit hit", state="error", expanded=False)
                return (
                    "Rate limit reached. You have exceeded the API quota for this minute. "
                    "Please wait 60 seconds and try again. If this keeps happening, "
                    "try reducing the number of active servers or shortening your query."
                ), input_tokens, output_tokens

            input_tokens  += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            text_parts = [b.text for b in response.content if b.type == "text"]

            if response.stop_reason == "end_turn":
                status.update(label="Done", state="complete", expanded=False)
                return "\n".join(text_parts) if text_parts else "(no response)", input_tokens, output_tokens

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                if text_parts:
                    status.write("Reasoning: " + " ".join(text_parts))
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    status.update(label="Calling " + block.name + "...")
                    with status.container():
                        with st.expander("Tool: " + block.name, expanded=False):
                            st.json(block.input)
                    if block.name not in tool_registry:
                        result_text = "Unknown tool: " + block.name
                    else:
                        session, orig_name = tool_registry[block.name]
                        try:
                            call_result = await session.call_tool(orig_name, block.input)
                            result_text = "\n".join(
                                c.text for c in call_result.content if hasattr(c, "text")
                            ) or "(empty result)"
                            preview = result_text[:300] + "..." if len(result_text) > 300 else result_text
                            status.write(block.name + " returned: " + preview)
                        except Exception as e:
                            result_text = "Tool error: " + str(e)
                            status.write(block.name + " error: " + str(e))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                status.update(label="Unexpected stop: " + response.stop_reason, state="error")
                return ("\n".join(text_parts) if text_parts else "Stopped: " + response.stop_reason), input_tokens, output_tokens


st.set_page_config(page_title="GitMCP Doc Assistant", layout="wide")
st.title("GitMCP Doc Assistant")
st.caption("Query GitHub-hosted documentation via gitmcp.io. Only gitmcp.io servers are permitted.")

with st.sidebar:
    st.header("Library Catalogue")
    st.markdown(
        "Tick up to **" + str(MAX_ACTIVE_SERVERS) + "** libraries to query at once. "
        "Selecting fewer reduces token usage and avoids rate limits."
    )

    # Count how many catalogue entries are currently checked
    catalogue_checked = sum(
        1 for lib in LIBRARY_CATALOGUE
        if lib["shortname"] in st.session_state.active_shortnames
    )
    at_limit = catalogue_checked >= MAX_ACTIVE_SERVERS

    for lang in ["Python", "R"]:
        libs = [l for l in LIBRARY_CATALOGUE if l["lang"] == lang]
        st.subheader(LANG_LABEL[lang])
        cols = st.columns(2)
        for idx, lib in enumerate(libs):
            col = cols[idx % 2]
            is_active = lib["shortname"] in st.session_state.active_shortnames
            # Disable unchecked boxes when at the limit
            disabled = at_limit and not is_active
            help_text = lib["desc"] + "\n\n" + gitmcp_url(lib["github"])
            if disabled:
                help_text = "Deselect another server first (max " + str(MAX_ACTIVE_SERVERS) + " active)"
            checked = col.checkbox(
                lib["shortname"],
                value=is_active,
                key="cb_" + lib["shortname"],
                help=help_text,
                disabled=disabled,
            )
            if checked:
                st.session_state.active_shortnames.add(lib["shortname"])
            else:
                st.session_state.active_shortnames.discard(lib["shortname"])

    if at_limit:
        st.warning("Max " + str(MAX_ACTIVE_SERVERS) + " servers selected. Deselect one to choose another.")

    st.divider()
    st.subheader("Custom GitMCP server")
    with st.expander("Example", expanded=False):
        st.markdown(
            "**Short name:** `pysparkdocs`  \n"
            "**URL:** `apache/spark`  \n\n"
            "Try asking: " + EXAMPLE_QUERY
        )

    custom_to_keep = []
    for i, cs in enumerate(st.session_state.custom_servers):
        c1, c2, c3 = st.columns([2, 4, 1])
        new_name = c1.text_input("Name", value=cs["shortname"], key="cname_" + str(i),
                                  label_visibility="collapsed", placeholder="shortname")
        new_url  = c2.text_input("URL",  value=cs["url"],       key="curl_"  + str(i),
                                  label_visibility="collapsed", placeholder="owner/repo")
        remove = c3.button("X", key="cdel_" + str(i))
        if not remove:
            custom_to_keep.append({"shortname": new_name, "url": new_url})
    st.session_state.custom_servers = custom_to_keep

    if st.button("Add custom server"):
        st.session_state.custom_servers.append({"shortname": "", "url": ""})
        st.rerun()

    st.divider()
    errors = []
    mcp_config = {}
    seen_names = set()

    for lib in LIBRARY_CATALOGUE:
        if lib["shortname"] in st.session_state.active_shortnames:
            mcp_config[lib["shortname"]] = gitmcp_url(lib["github"])
            seen_names.add(lib["shortname"])

    for cs in st.session_state.custom_servers:
        name = cs["shortname"].strip()
        raw  = cs["url"].strip()
        if not name and not raw:
            continue
        if not name:
            errors.append("A custom server is missing a short name.")
            continue
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            errors.append(name + ": use only letters, digits, underscores.")
            continue
        if name in seen_names:
            errors.append("Duplicate short name: " + name)
            continue
        resolved = normalise_gitmcp_url(raw)
        if not is_valid_gitmcp_url(resolved):
            errors.append(name + ": must be a gitmcp.io/owner/repo URL.")
            continue
        seen_names.add(name)
        mcp_config[name] = resolved

    for e in errors:
        st.warning(e)

    if mcp_config:
        st.success(str(len(mcp_config)) + " server(s) configured")
    else:
        st.info("Select at least one library to start.")

    st.divider()
    st.subheader("Session cost")
    total_cost = calc_cost(st.session_state.total_input_tokens, st.session_state.total_output_tokens)
    st.metric("Total cost", "$" + "{:.6f}".format(total_cost))
    c1, c2 = st.columns(2)
    c1.metric("Input tokens",  "{:,}".format(st.session_state.total_input_tokens))
    c2.metric("Output tokens", "{:,}".format(st.session_state.total_output_tokens))
    st.caption("Pricing: $1 / $5 per 1M tokens (claude-haiku-4-5)")

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.session_state.total_input_tokens  = 0
        st.session_state.total_output_tokens = 0
        st.rerun()


if not mcp_config:
    st.warning("Select at least one library from the sidebar to start chatting.")
    st.stop()

prev_config_key = str(sorted(mcp_config.items()))
if st.session_state.connection_status.get("_config_key") != prev_config_key:
    with st.status("Probing servers...", expanded=True) as probe_status:
        probe_results = asyncio.run(probe_servers(mcp_config))
        st.session_state.connection_status = probe_results
        st.session_state.connection_status["_config_key"] = prev_config_key
        probe_status.update(label="Server probe complete", state="complete", expanded=False)

conn = st.session_state.connection_status

with st.expander("Connected servers", expanded=True):
    catalogue_lookup = {l["shortname"]: l for l in LIBRARY_CATALOGUE}
    col_headers = st.columns([2, 1, 1, 2])
    col_headers[0].markdown("**Server**")
    col_headers[1].markdown("**Status**")
    col_headers[2].markdown("**Tools**")
    col_headers[3].markdown("**Doc version**")
    st.divider()
    for name in mcp_config:
        info     = conn.get(name, {})
        ok       = info.get("ok", False)
        n_tools  = info.get("tools", 0)
        version  = info.get("version", None)
        error    = info.get("error", None)
        lib_meta = catalogue_lookup.get(name)
        label    = name + (" (" + lib_meta["desc"] + ")" if lib_meta else "")
        version_str = version if version else "unknown"
        cols = st.columns([2, 1, 1, 2])
        cols[0].markdown(label)
        cols[1].markdown("Connected" if ok else "Failed")
        cols[2].markdown(str(n_tools) if ok else "-")
        cols[3].markdown("`" + version_str + "`" if ok else (error or "-")[:60])


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "cost" in msg:
            st.caption(
                "This response: $" + "{:.6f}".format(msg["cost"])
                + "  |  in: " + str(msg["input_tokens"]) + " tokens"
                + "  out: " + str(msg["output_tokens"]) + " tokens"
            )

shown = list(mcp_config.keys())[:4]
suffix = "..." if len(mcp_config) > 4 else ""
placeholder = "Ask about " + ", ".join(shown) + suffix

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status("Starting...", expanded=True) as status:
            answer, in_tok, out_tok = asyncio.run(run_conversation(prompt, mcp_config, status))
        st.markdown(answer)
        msg_cost = calc_cost(in_tok, out_tok)
        st.caption(
            "This response: $" + "{:.6f}".format(msg_cost)
            + "  |  in: " + str(in_tok) + " tokens"
            + "  out: " + str(out_tok) + " tokens"
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cost": msg_cost,
        })
        st.session_state.total_input_tokens  += in_tok
        st.session_state.total_output_tokens += out_tok
        st.rerun()
