from datetime import date
import streamlit as st
import pandas as pd
import urllib
import pyodbc
import re
from sqlalchemy import create_engine, text
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import socket
import os
import time
import json
from cryptography.fernet import Fernet
import secrets
import streamlit.components.v1 as components
import requests
from datetime import datetime


genai.configure(api_key="AIzaSyD09VntQLgT1aPBbZOG6s4zCUnbyREx2Bg")
model = genai.GenerativeModel("gemini-2.5-flash")

# 🔹 Streamlit config
st.set_page_config("DataGenie", layout="wide",
                   initial_sidebar_state="expanded")

# ssms_servers = [
#     {
#         "name": "COSMOS\\SQLEXPRESS01",
#         "server": "COSMOS\\SQLEXPRESS01,53112",  # dynamically set IP
#         "username": "sa",
#         "password": "abcd123456"
#     }
# ]

# server_cfg = ssms_servers[0]

# # Build connection string safely
# params = urllib.parse.quote_plus(
#     f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#     f"SERVER={server_cfg['server']};"
#     f"DATABASE=query_genie;"
#     f"UID={server_cfg['username']};"
#     f"PWD={server_cfg['password']};"
# )

# # SQLAlchemy engine (single connection for everything)
# engine = create_engine(
#     f"mssql+pyodbc:///?odbc_connect={params}", fast_executemany=True)

ssms_servers = [
    {
        "name": "EC2_SQLSERVER",
        "server": "localhost,1433",       # dynamically set IP
        "username": "SA",
        "password": "abcd@123456"
    }
]

server_cfg = ssms_servers[0]

# Build connection string safely
params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={server_cfg['server']};"
    f"DATABASE=query_genie;"
    f"UID={server_cfg['username']};"
    f"PWD={server_cfg['password']};"
)

# SQLAlchemy engine (single connection for everything)
engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}",
    fast_executemany=True,
    pool_pre_ping=True,      # 🩵 auto check connection health
    pool_recycle=1800,       # 🔁 refresh every 30 mins
    pool_timeout=30          # ⏱️ don’t hang forever
)
# ----------------- Encryption Key -----------------
fernet_key = b'Sv_cBtT5H5i_fv3sPvRrAe_2z6WRnqbmq-rmfxUyiGQ='
cipher_suite = Fernet(fernet_key)
RECAPTCHA_SECRET_KEY = "6LfkXZQrAAAAAKIosm2eIEKwzw6AmblfqY8NDb3D"   # from Google
RECAPTCHA_SITE_KEY = "6LfkXZQrAAAAANLCHFVeHYym1YO0F_6aa9mcbziC"


def get_user_credentials():
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT username, password FROM dbo.login_credentials"))
        rows = result.fetchall()
        return {row[0]: row[1] for row in rows}


def verify_recaptcha(token):
    url = "https://www.google.com/recaptcha/api/siteverify"
    response = requests.post(url, data={
        "secret": RECAPTCHA_SECRET_KEY,
        "response": token
    })
    return response.json().get("success", False)

# -------------------- SHOW VISIBLE CAPTCHA --------------------


def show_recaptcha():
    """Display Google reCAPTCHA v2 checkbox."""
    recaptcha_html = f"""
    <script src="https://www.google.com/recaptcha/api.js" async defer></script>
    <div class="g-recaptcha" data-sitekey="{RECAPTCHA_SITE_KEY}"></div>
    """
    components.html(recaptcha_html, height=100)


def login_page():

    col1, col2, col3 = st.columns([3, 2, 3])
    with col1:
        st.image("techfer_logo_new.png", width=150,)
    with col2:
        st.header("👨‍💻 Login ")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("🔑 Login"):

            # Get token from client via JS (Streamlit cannot auto-read checkbox token, so use hidden input workaround)
            # recaptcha_token = st.experimental_get_query_params().get(
            #     "g-recaptcha-response", [""])[0]
            # if not recaptcha_token or not verify_recaptcha(recaptcha_token):
            #     st.error("⚠️ Please complete the reCAPTCHA")
            #     return

            credentials = get_user_credentials()
            if username in credentials and credentials[username] == password:
                st.success(f"✅ Welcome, {username}!")

                # Record login
                with engine.begin() as conn:
                    print("Engine Begin")
                    conn.execute(
                        text(
                            "INSERT INTO dbo.login_tracker (username, loginTime) VALUES (:username, GETDATE())"
                        ),
                        {"username": username}
                    )
                    print("complete")
                token = secrets.token_hex(16)
                data = json.dumps(
                    {"username": username, "token": token}).encode()
                encrypted_data = cipher_suite.encrypt(data).decode()

                st.session_state.update({
                    "username": username,
                    "authenticated": True,
                    "last_activity": time.time(),
                    "encrypted_token": encrypted_data,
                    "page": "landing"
                })

                st.rerun()
            else:
                st.error("❌ Invalid username or password")


def landing_page():
    global engine, server_cfg
    ssms_schema_df = pd.DataFrame()
    m_p = st.empty()

    timeout_seconds = 1200  # 5 minutes

    last_activity = st.session_state.get("last_activity", time.time())

    if time.time() - last_activity > timeout_seconds:
        placeholder = st.empty()  # placeholder for warning
        placeholder.warning("⚠️ Session expired due to inactivity.")
        time.sleep(3)  # show warning for 3 seconds
        placeholder.empty()

        # ✅ Update logout time in DB before clearing session
        username_for_db = st.session_state.get("username", "")
        if username_for_db:
            try:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE dbo.login_tracker
                        SET logoutTime = GETDATE()
                        WHERE username = :username
                        AND loginTime = (
                            SELECT MAX(loginTime)
                            FROM dbo.login_tracker
                            WHERE username = :username
                        )
                    """), {"username": username_for_db})
            except Exception as e:
                st.warning(f"⚠️ Could not update logout time: {e}")
                print("error",e)
        # Clear session and redirect
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.rerun()

    # Update last_activity on every rerun
    st.session_state["last_activity"] = time.time()

    conn_str = (
        f"""mssql+pyodbc://{server_cfg['username']}:{server_cfg['password']}"""
        f"""@{server_cfg['server']}/query_genie?driver=ODBC+Driver+17+for+SQL+Server""")

    engine = create_engine(conn_str, fast_executemany=True)

    # Optional: wider sidebar
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            max-width: 1000px;
            min-width: 600px;
            overflow-x: auto;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-right: 1rem;
        }

        .canvas-box {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            min-height: 400px;
        }
        </style>
    """, unsafe_allow_html=True)

    # # --- Handle not logged in initially ---
    # if not username and not st.session_state.logged_out:
    #     st.error("Please login via Flask first.")
    #     st.stop()

    chat_file = f"chat_history_{date.today()}.json"

    def load_chat_history():
        if os.path.exists(chat_file):
            with open(chat_file, "r") as f:
                return json.load(f)
        return {}

    def save_chat_history(history):
        with open(chat_file, "w") as f:
            json.dump(history, f, indent=4)

    # Load existing history
    history = load_chat_history()

    # Ensure current user has a list
    current_user = st.session_state.get("username", "")
    if current_user and current_user not in history:
        history[current_user] = []   # ✅ fixed here

    # Session States
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_result_df" not in st.session_state:
        st.session_state.query_result_df = pd.DataFrame()
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_query_columns" not in st.session_state:
        st.session_state.last_query_columns = []

    # 🔹 SQL Server Config
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    @st.cache_data(show_spinner=False)
    def fetch_ssms_schema():
        data = []
        for s in ssms_servers:
            try:
                dbs = ["AdventureWorks2022"]  # ✅ Only fetch AdventureWorks

                for db in dbs:
                    db_conn = f"""Driver={{ODBC Driver 17 for SQL Server}};Server={
                        s['server']};Database={db};UID={
                        s['username']};PWD={
                        s['password']};Encrypt=no;"""
                    engine = create_engine(
                        f"""mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(db_conn)}""")
                    df = pd.read_sql(
                        "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS",
                        engine
                    )
                    df['SERVER'] = s['name']
                    df['DATABASE'] = db
                    data.append(df)
            except Exception as e:
                st.warning(f"SSMS Error: {e}")
        return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

    def build_chat_context():
        conversation = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                conversation.append(f"User: {msg['message']}")
            elif msg["role"] == "assistant":
                conversation.append(f"Assistant: {msg['message']}")
        return "\n".join(conversation)

    def gen_join_queries(user_input, ssms_schema, history=""):
        user_input_add = user_input.strip().rstrip('.') + " from SSMS server"
        history_text = f"Conversation so far:\n{history}\n\n"
        prompt = f"""
        Their previous chat was:{history_text}

        Current user question:  "{user_input_add}"
        SSMS Schema: {ssms_schema}
        Understand the Histry and user_input_add ,on this based understand the need and follow the Following instruction:
            You are a smart SQL developer but you have to follow the following guidelines only:
                - You are given the schema of one SQL Server (SSMS).
                - Write one SQL query using SSMS schema only to fetch data requested in the user question.
                - Do not rename, infer, or substitute any column.
                - Use only fully qualified table and column names exactly as given.
                - strictly Use database name to write the whole query including select statement.
                - strictly Use this format to write query: DATABASE.SCHEMA.TABLE.COLUMN
                - Strictly use a column only once in a wuery if used more than once give it alias.
                - Use Syntax which support ssms
                - Label the query using:
                    -- SSMS Query Start
                    <SQL>
                """
        return model.generate_content(prompt).text

    def detect_mode(user_text):
        classification_prompt = f"""
        You are an intent classifier.
        The user said: "{user_text}"

        Classify the intent into one of the following categories:

        - "Query": When the user asks for raw data, SQL, data fetching, tables, aggregations or database queries.
        - "Descriptive": When the user wants summaries, facts, trends, or general descriptions about what the data shows.
        - "Diagnostic": When the user wants explanations or reasons behind data trends or results.
        - "Predictive": When the user asks for forecasts or predictions based on data.
        - "Prescriptive": When the user wants actionable suggestions, decisions, or recommendations based on the data.

        Examples:
        - "Get total sales by year" → Query
        - "Summarize the yearly sales trend" → Descriptive
        - "Why did sales drop in 2019?" → Diagnostic
        - "Predict 2025 revenue using this data" → Predictive
        - "What should we focus on next year to increase sales?" → Prescriptive
        DO NOT GIVE PYTHON CODE
        Output ONLY one of the following words: Query, Descriptive, Diagnostic, Predictive, Prescriptive.
        """
        response = model.generate_content(classification_prompt).text.strip()
        return response

    history_text = build_chat_context()

    # 🔹 Fetch Schema Once
    if ssms_schema_df.empty:
        with st.spinner("Fetching SSMS schema..."):
            ssms_schema_df = fetch_ssms_schema()

    # 🔹 Logo
    st.image("techfer_logo_new.png", width=200,)

    with st.sidebar:
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background-color: white;
                color: black;
                border-radius: 8px;
                padding: 0.4em 1.5em;
                font-size: 14px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            div.stButton > button:first-child:hover {
                background-color: #FF6B6B;
                color: white;
                box-shadow: 0 0 10px rgba(255, 75, 75, 0.6);
                transform: scale(1.05);
            }
            </style>
        """, unsafe_allow_html=True)

        if st.session_state.get("username"):
            if st.button("🚪 Logout"):
                username_for_db = st.session_state.username
                try:
                    with engine.begin() as conn:
                        conn.execute(text("""
                            WITH LatestLogin AS (
                                SELECT TOP 1 *
                                FROM dbo.login_tracker
                                WHERE username = :username
                                ORDER BY loginTime DESC
                            )
                            UPDATE LatestLogin
                            SET logoutTime = GETDATE()
                        """), {"username": username_for_db})

                except Exception as e:
                    #st.warning(f"⚠️ Could not update logout time: {e}")
                    print("error: ",e)
                # ✅ Clear session state completely
                st.session_state.clear()

                # ✅ Set page before rerun to ensure redirect
                st.session_state["page"] = "login"

                # 3-2-1 Countdown (optional)
                countdown_placeholder = st.empty()
                for i in range(3, 0, -1):
                    countdown_placeholder.warning(f"⚡ Logging out in {i}...")
                    time.sleep(1)
                countdown_placeholder.empty()

                # ✅ Rerun to show login page
                st.rerun()

        # -------------------- App UI --------------------
        st.markdown("""
            <h1 style="font-size: 35px; color: #2C3E50; margin-top: -40px; text-align: center;">
                DataGenie
            </h1>
        """, unsafe_allow_html=True)

        # ✅ Initialize session variables safely
        for key, default in {
            "query_result_df": pd.DataFrame(),
            "chat_history": [],
            "last_query": "",
            "last_query_columns": []
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default

        # ✅ Handle login status
        if "username" in st.session_state and st.session_state["username"]:
            st.success(f"👋 Welcome, {st.session_state.username}")
            m_p.empty()
        else:
            st.warning(
                "⚠️ You are not logged in. Please login via the login page.")

            # ✅ Safer redirect (within Streamlit, not Flask)
            st.session_state["page"] = "login"
            st.rerun()

        # ✅ Show data if available
        if not st.session_state.query_result_df.empty:
            new_df1 = st.session_state.query_result_df.copy()
            new_df1.reset_index(drop=True, inplace=True)
            new_df1.index = new_df1.index + 1

            # === Initialize States ===
            if "active_tab" not in st.session_state:
                st.session_state.active_tab = "data"
            if "chart_metadata" not in st.session_state:
                st.session_state["chart_metadata"] = []

            # === Tab Buttons ===
            colA, colB = st.columns([1, 4])
            with colA:
                if st.button("📊 Data"):
                    st.session_state.active_tab = "data"
            with colB:
                if st.button("📈 Visualize"):
                    st.session_state.active_tab = "viz"

            # === Data Tab ===
            if st.session_state.active_tab == "data":
                st.dataframe(new_df1)
                with st.expander("Show Query", expanded=False):
                    st.code(st.session_state.last_query)
            else:
                st.info("Please request data to generate chart!!")

            # === Visualization Tab ===
            if st.session_state.active_tab == "viz":

                if "last_df_shape" not in st.session_state or st.session_state["last_df_shape"] != new_df1.shape:
                    st.session_state.pop("generated_chart_code", None)
                    st.session_state["last_df_shape"] = new_df1.shape

                x_axis_cols = st.multiselect(
                    "📌 Select X-axis columns",
                    new_df1.columns.tolist(),
                    default=st.session_state.get("x_axis_cols", [])
                )
                y_axis_cols = st.multiselect(
                    "📌 Select Y-axis columns",
                    new_df1.columns.tolist(),
                    default=st.session_state.get("y_axis_cols", [])
                )

                chart_prompt = st.text_area(
                    "📝 Describe the chart you want to generate",
                    value=st.session_state.get("chart_prompt", "")
                )

                st.subheader("📈 Gemini Chart Canvas")

                if st.button("🎨 Create Chart"):
                    if not x_axis_cols or not y_axis_cols or not chart_prompt:
                        st.warning(
                            "Select X & Y columns and enter chart description.")
                    else:
                        st.session_state["x_axis_cols"] = x_axis_cols
                        st.session_state["y_axis_cols"] = y_axis_cols
                        st.session_state["chart_prompt"] = chart_prompt

                        x_list = ", ".join(x_axis_cols)
                        y_list = ", ".join(y_axis_cols)

                        chart_gen_prompt = f"""
                                You are a Python data visualization assistant.

                                The user wants a chart based on this request: {chart_prompt}

                                Selected columns from the DataFrame named `df`:
                                - X-axis: {x_list}
                                - Y-axis: {y_list}

                                Instructions:
                                - Use the existing DataFrame `df` as-is. Do not create or redefine `df` or generate any mock/sample data.
                                - Use Plotly Express or Plotly Graph Objects.
                                - If widgets are selected, integrate them.
                                - Before plotting, drop any rows where required columns (like X, Y, hierarchy path, or value columns) are null, NaN, or blank strings ('').
                                - Output only the Python code inside a markdown code block.
                                """

                        response = model.generate_content(
                            chart_gen_prompt).text
                        print("chart code : ", response)
                        chart_code = re.search(
                            r"```python(.*?)```", response, re.DOTALL)

                        if chart_code:
                            st.session_state["generated_chart_code"] = chart_code.group(
                                1).strip()
                        else:
                            st.error("⚠️ Couldn't parse chart code.")
                            time.sleep(5)
                            m_p.empty()

                # === Create & Store Charts ===
                if "generated_chart_code" in st.session_state:
                    try:
                        exec_globals = {"pd": pd, "df": new_df1,
                                        "px": px, "go": go, "np": np}
                        exec(
                            st.session_state["generated_chart_code"], exec_globals)

                        new_figs = [
                            exec_globals[name]
                            for name in exec_globals
                            if re.match(r"fig\d*$", name) and isinstance(exec_globals[name], go.Figure)
                        ]

                        if new_figs:
                            for fig in new_figs:
                                # ✅ Add this to render new chart
                                st.plotly_chart(fig, use_container_width=True)

                            st.session_state["chart_metadata"].append({
                                "code": st.session_state["generated_chart_code"],
                                "x_cols": x_axis_cols,
                                "y_cols": y_axis_cols
                            })

                        st.session_state.pop("generated_chart_code", None)

                    except Exception as e:
                        st.error("❌ Chart rendering failed.")
                        time.sleep(5)
                        m_p.empty()
                        st.exception(e)
                        # time.sleep(5)
                        # m_p.empty()

                if st.session_state["chart_metadata"]:
                    st.subheader("📊 Created Charts")

                    # Column Filters
                    df = st.session_state.query_result_df
                    filter_cols = st.multiselect(
                        "Select columns to filter", df.columns.tolist()
                    )

                    filters = {}
                    for col in filter_cols:
                        unique_vals = sorted(df[col].dropna().unique())
                        selected_vals = st.multiselect(
                            f"Filter {col}", unique_vals, default=unique_vals
                        )
                        filters[col] = selected_vals

                    # Apply filters
                    filtered_df = df.copy()
                    for col, vals in filters.items():
                        filtered_df = filtered_df[filtered_df[col].isin(vals)]

                    # Display last 6 charts using filtered_df as df
                    grid_cols = st.columns(3)
                    # Directly loop over the last 6 chart entries with their true indices
                    for display_i, chart_index in enumerate(range(max(0, len(st.session_state["chart_metadata"]) - 6), len(st.session_state["chart_metadata"]))):
                        meta = st.session_state["chart_metadata"][chart_index]
                        exec_globals = {"pd": pd, "df": filtered_df,
                                        "px": px, "go": go, "np": np}
                        exec(meta["code"], exec_globals)
                        fig = next(v for v in exec_globals.values()
                                   if isinstance(v, go.Figure))

                        with grid_cols[display_i % 3]:
                            delete_key = f"delete_chart_{chart_index}"
                            if st.button("❌", key=delete_key):
                                st.session_state["chart_metadata"].pop(
                                    chart_index)
                                st.rerun()

                            st.plotly_chart(fig, use_container_width=True,
                                            key=f"chart_{display_i}")
                else:
                    st.info(
                        "No chart generated yet. Use the controls above to create one.")

    def sanitize_gemini_response(text):
        text = re.sub(r'</?div[^>]*>', '', text)
        return text.strip()

    for msg in st.session_state.chat_history:
        if msg["role"] == "separator":
            st.markdown("<hr style='border: 1px solid #ccc;'>",
                        unsafe_allow_html=True)
        elif msg["role"] == "assistant":
            clean_text = sanitize_gemini_response(msg['message'])
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; margin-bottom: 20px;'>
                <div style='background-color: #e6f3ff; padding: 10px 14px; border-radius: 15px 15px 15px 0; max-width: 80%; white-space: normal; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    {clean_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif msg["role"] == "user":
            clean_text = sanitize_gemini_response(msg['message'])
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                <div style='background-color: #E8E8E8; padding: 10px 14px; border-radius: 15px 15px 0 15px; max-width: 80%; white-space: normal; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    {clean_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # --- Bottom Chat Input ---
    user_input = st.chat_input("Type your question...")

    if user_input:
        mode = detect_mode(user_input)   # <-- Auto mode detection here
        print(mode)
        if mode == "Query":
            schema_text = re.sub(
                r'\s{2,}', ' ', ssms_schema_df.to_string(index=False).strip())
            sql_text = gen_join_queries(
                user_input, schema_text, history_text)
            # print("sql query:", sql_text)
            cleaned_output = sql_text.replace("sql", "").strip()
            match = re.search(r"--\s*SSMS Query Start\s*(.*)",
                              cleaned_output, re.DOTALL | re.IGNORECASE)

            if match:
                query = match.group(1).strip("`").rstrip(";").strip()
                st.session_state.last_query = query

                st.session_state.chat_history.append(
                    {"role": "separator", "message": "---"})
                st.session_state.chat_history.append(
                    {"role": "user", "message": user_input})

                server_cfg = ssms_servers[0]
                conn_str = (
                    f"""Driver={{ODBC Driver 17 for SQL Server}};"""
                    f"""Server={server_cfg['server']};"""
                    f"""UID={server_cfg['username']};"""
                    f"""PWD={server_cfg['password']};"""
                    f"""Encrypt=no;TrustServerCertificate=yes;"""
                )
                quoted_conn = urllib.parse.quote_plus(conn_str)
                engine = create_engine(
                    f"""mssql+pyodbc:///?odbc_connect={quoted_conn}""")
                df = pd.read_sql(query, engine)
                df.fillna("", inplace=True)
                st.session_state.query_result_df = df
                if df.empty:
                    st.warning("Query returned no data")
                else:
                    st.dataframe(df)

                print(df.head())
                # Save query to JSON history
                current_user = st.session_state.get("username", "")
                if current_user:
                    history[current_user].append(
                        {"role": "user", "message": user_input})
                    history[current_user].append(
                        {"role": "assistant",
                            "message": query}
                    )
                    save_chat_history(history)

                else:
                    if current_user:
                        history[current_user].append(
                            {"role": "user", "message": user_input})
                        history[current_user].append(
                            {"role": "assistant",
                                "message": "❌ Could not parse valid SQL query."}
                        )
                        save_chat_history(history)

                        st.error("❌ Could not parse valid SQL query.")
                        time.sleep(5)
        else:
            # === CHAT MODE ===
            st.session_state.chat_history.append(
                {"role": "user", "message": user_input}
            )

            # ✅ Build conversation history for context

            print("""
                    show me :

                """, history_text)
            if not st.session_state.query_result_df.empty:
                df = st.session_state.query_result_df
                prompt = f"""
                Conversation so far:
                {history_text}

                    You are a data consultant. Your role is to analyze all the given data and answer the user's question with clear, actionable insights.
                    - Do not give python code
                    - If Asked for any kind of calculation, consider the whole data and give the answer
                    User question: "{user_input}"
                    Data:
                    {df.to_markdown(index=True)}

                    Respond in bullet points with clear insights.
                """
            else:
                prompt = f"""
                    Conversation so far:
                    {history_text}

                    You are a domain expert in all fields and Data Consultant expert.
                    User asked: \"{user_input}\"
                    Respond in 4-5 bullet points with useful analysis/suggestions.
                    """

            reply = model.generate_content(prompt).text
            st.session_state.chat_history.append(
                {"role": "assistant", "message": reply}
            )
            if current_user:
                history[current_user].append(
                    {"role": "user", "message": user_input})
                history[current_user].append(
                    {"role": "assistant", "message": reply})
                save_chat_history(history)

        st.rerun()


if "page" not in st.session_state:
    st.session_state["page"] = "login"

if st.session_state["page"] == "login":
    login_page()
    st.stop()  # Don't run landing page until login
elif st.session_state["page"] == "landing":
    if not st.session_state.get("authenticated", False):
        st.session_state["page"] = "login"
        st.rerun()
    else:
        landing_page()  # ✅ call landing page here


