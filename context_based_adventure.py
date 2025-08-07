import streamlit as st
import pandas as pd
import urllib
import pyodbc
import re
from sqlalchemy import create_engine
import google.generativeai as genai
import os
import json
import html
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import socket
import requests
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Load session token securely
if not os.path.exists("session_token.json"):
    st.error("❌ No active session. Please log in.")
    st.stop()

with open("session_token.json", "r") as f:
    session_data = json.load(f)

encrypted_data = session_data.get('encrypted_data', None)

# Your Fernet key (should match the one used in login.py)
fernet_key = b'Sv_cBtT5H5i_fv3sPvRrAe_2z6WRnqbmq-rmfxUyiGQ='
cipher_suite = Fernet(fernet_key)

try:
    # Decrypt and load session info
    decrypted_text = cipher_suite.decrypt(encrypted_data.encode()).decode()
    session_info = json.loads(decrypted_text)

    username = session_info.get("username")
    token = session_info.get("token")

    # st.success(f"✅ Welcome, {username}")

    # Optionally delete session file after successful load
   # os.remove("session_token.json")

except Exception as e:
    st.error("❌ Decryption failed. Invalid or tampered token.")
    st.stop()


load_dotenv(dotenv_path="key.env")

# Fetch the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")
print(api_key)
# Configure Gemini

def get_public_ip():
    return requests.get('https://api.ipify.org').text

public_ip = get_public_ip()
print(public_ip)

# Configure Gemini
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
st.set_page_config("DataGenie", layout="wide")
#hide_streamlit_style = """
 #   <style>
   # #MainMenu {visibility: hidden;}
  #  footer {visibility: hidden;}
 #   header {visibility: hidden;}
#    </style>
#"""

#st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "query_result_df" not in st.session_state:
    st.session_state.query_result_df = pd.DataFrame()
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

ssms_servers = [{
    "name": "EC2_SQLSERVER",   # or just "SQLEXPRESS"
    "server": "localhost,1433",       # dynamically set IP
    "username": "SA",
    "password": "Admin@1234"
}]


# Global DataFrames
ssms_schema_df = pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_ssms_schema():
    data = []
    for s in ssms_servers:
        try:
            base_conn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={s['server']};UID={s['username']};PWD={s['password']};"
            with pyodbc.connect(base_conn) as c:
                cursor = c.cursor()
                print("cursor")
                cursor.execute(
                    "SELECT name FROM sys.databases WHERE name NOT IN ('master','tempdb','model','msdb')"
                )
                print('cursor executed')
                dbs = [r[0] for r in cursor.fetchall()]
                for db in dbs:
                    # ✅ Build correct connection string for specific database
                    db_conn = f"Driver={{ODBC Driver 17 for SQL Server}};Server={s['server']};Database={db};UID={s['username']};PWD={s['password']}"
                    engine = create_engine(
                        f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(db_conn)}"
                    )
                    df = pd.read_sql(
                        "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS",
                        engine
                    )
                    
                    df['SERVER'] = s['name']
                    df['DATABASE'] = 'AdventureWorks2022'  # ✅ use actual db name
                    data.append(df)
                   
        except Exception as e:
            st.warning(f"Something went wrong,please try again")
	   # print("SSMS Error: ",e)
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()


def gen_join_queries(user_input, ssms_schema):
    user_input_add = user_input.strip().rstrip('.') + " from SSMS server"
    prompt = f"""
User question: "{user_input_add}"

SSMS Schema: {ssms_schema}

- You are a smart SQL developer but you have to follow the following guidelines only:
    - You are given the schema of one SQL Server (SSMS).
    - Write one SQL query using SSMS schema only to fetch data requested in the user question.
    - Do not rename, infer, or substitute any column.
    - If a field in the user question doesn't exist in the schema, skip it entirely.
    - Use only fully qualified table and column names exactly as given.
    - strictly Use database name to write the whole query including select statement.
    - strictly Use this format to write query: DATABASE.SCHEMA.TABLE.COLUMN (uppercase preferred), for the whole query including select statement.
    - If two or more columns have the same name (e.g., Name, ID, Price), then assign each a unique alias using this format:
        DATABASE.SCHEMA.TABLE.COLUMN AS [TABLE_COLUMN] 
    - Use Syntax which support ssms
    - Label the query using:
        -- SSMS Query Start
        <SQL>
    """
    return model.generate_content(prompt).text

if ssms_schema_df.empty:
    with st.spinner("Installing dependencies..."):
        ssms_schema_df = fetch_ssms_schema()
    # print(ssms_schema_df)
    print(ssms_schema_df)

st.image('techfer_logo_new.png',width=150)
# Sidebar - SQL input and results
with st.sidebar:
    st.title("DataGenie  ") 

    nl_query = st.text_area("Enter your question ")
    
    if st.button("Generate        "):
        if nl_query:
            schema_text = re.sub(r'\s{2,}', ' ', ssms_schema_df.to_string(index=False).strip())
            sql_text = gen_join_queries(nl_query, schema_text)
            cleaned_output = sql_text.replace("sql", "").replace("", "").strip()
            match = re.search(r"--\s*SSMS Query Start\s*(.*)", cleaned_output, re.DOTALL | re.IGNORECASE)

            if match:
                query = match.group(1).strip("`").rstrip(";").strip()
                st.session_state.last_query = query

                # Append to chat history
                st.session_state.chat_history.append({"role": "separator", "message": "---"})
                st.session_state.chat_history.append({"role": "user", "message": nl_query})

                server_cfg = ssms_servers[0]
                conn_str = (
                    f"Driver={{ODBC Driver 17 for SQL Server}};"
                    f"Server={server_cfg['server']};"
                    f"UID={server_cfg['username']};"
                    f"PWD={server_cfg['password']};"
                )
                quoted_conn = urllib.parse.quote_plus(conn_str)
                engine = create_engine(f"mssql+pyodbc:///?odbc_connect={quoted_conn}")
                df = pd.read_sql(query, engine)
                st.session_state.query_result_df = df
                st.success("✅ Data fetched and ready to analyze")
            else:
                st.error("❌ Could not parse valid SQL query.")

    if st.session_state.last_query:
        with st.expander("📎Query       "):
            st.code(st.session_state.last_query, language="sql")

    if not st.session_state.query_result_df.empty:
        with st.expander("📊 Data        "):
            st.dataframe(st.session_state.query_result_df)

        st.markdown("### 📈 Visualize        ")

        new_df1 = st.session_state.query_result_df

        # Reset chart state if data shape changed
        if "last_df_shape" not in st.session_state or st.session_state["last_df_shape"] != new_df1.shape:
            st.session_state.accepted_charts = []
            st.session_state.pop("generated_chart_code", None)
            st.session_state["last_df_shape"] = new_df1.shape

        if "accepted_charts" not in st.session_state:
            st.session_state.accepted_charts = []

        selected_cols = st.multiselect("📌 Select columns", new_df1.columns.tolist())
        chart_prompt = st.text_area("📝 Describe the chart you want to generate")

        if st.button("🎨 Create       "):
            if not selected_cols or not chart_prompt:
                st.warning("Select columns and enter chart description.")
            else:
                column_list = ", ".join(selected_cols)
                chart_gen_prompt = f"""
                    You are a Python data visualization assistant.

                    The user wants a chart based on this request: {chart_prompt}
                    Selected columns from the DataFrame named `df`: {column_list}

                    Instructions:
                    - Use the DataFrame `df` directly (do not create or redefine it).
                    - If the chart requires aggregation (like sum, count, average, etc.), use `groupby()` with appropriate aggregation (`sum()`, `count()`, `mean()`, etc.) based on user intent.
                    - For example: 
                    grouped_df = df.groupby(['column1'], as_index=False)['column2'].sum()
                    Then use grouped_df in the chart.
                    - If no aggregation is needed, use `df` directly.
                    - If multiple charts are requested, generate up to 4 figures: `fig1`, `fig2`, `fig3`, `fig4`.
                    - Use Plotly Express or Plotly Graph Objects.
                    - Do not include any comments, explanations, or data creation — just output pure valid Python code using the given `df`.

                    Output only the Python code inside a markdown code block (```python ... ```).
                    """

                response = model.generate_content(chart_gen_prompt).text
                chart_code = re.search(r"```python(.*?)```", response, re.DOTALL)
                if chart_code:
                    st.session_state["generated_chart_code"] = chart_code.group(1).strip()
                else:
                    st.error("⚠️ Couldn't parse chart code.")

        # Render chart preview
        if "generated_chart_code" in st.session_state:
            try:
                exec_globals = {
                    "pd": pd, "df": new_df1, "px": px, "go": go, "np": np
                }
                exec(st.session_state["generated_chart_code"], exec_globals)
                for name in exec_globals:
                    if re.match(r"fig\d*$", name) and isinstance(exec_globals[name], go.Figure):
                        fig = exec_globals[name]
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error("❌ Chart rendering failed.")
               # st.exception(e)
    else:
        st.info("Please request data to generate a chart.")

# Main chat UI

# st.markdown("### 💬 Conversation")

def sanitize_gemini_response(text):
    # Remove outer div tags, script/style tags, and redundant whitespace
    text = re.sub(r'</?div[^>]*>', '', text)           # Remove <div> and </div>
    text = text.strip()
    return text

for msg in st.session_state.chat_history:
    if msg["role"] == "separator":
        st.markdown("<hr style='border: 1px solid #ccc;'>", unsafe_allow_html=True)

    elif msg["role"] == "assistant":
        clean_text =sanitize_gemini_response(msg['message'])
        st.markdown(f"""
        <div style='
            display: flex;
            justify-content: flex-start;
            margin-bottom: 20px;
        '>
            <div style='
                background-color: #e6f3ff;
                padding: 10px 14px;
                border-radius: 15px 15px 15px 0;
                max-width: 80%;
                width: fit-content;
                white-space: pre-wrap;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            '>{clean_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif msg["role"] == "user":
        clean_text =sanitize_gemini_response(msg['message'])
        st.markdown(f"""
        <div style='
            display: flex;
            justify-content: flex-end;
            margin-bottom: 10px;
        '>
            <div style='
                background-color: #E8E8E8;
                padding: 10px 14px;
                border-radius: 15px 15px 0 15px;
                max-width: 80%;
                width: fit-content;
                white-space: pre-wrap;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            '>{clean_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
user_input = st.chat_input("Ask something... e.g., How can I improve retention?")
if user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})
    
    # Prepare context-aware prompt
    if not st.session_state.query_result_df.empty:
        df = st.session_state.query_result_df
        prompt = f"""
You are a data analyst. Your role is to analyze the given data and answer the user's question with clear, actionable insights.
Use the following structure based on the type of question:

User question: "{user_input}"
Data:
{df.to_markdown(index=False)}

Response Guidelines:
-If the question is Descriptive ("What happened?"):
Summarize key trends, patterns, or changes in the data.

Highlight important metrics (e.g., increase/decrease in values).

Keep it factual and to the point.

-If the question is Diagnostic ("Why did it happen?"):
Identify root causes, anomalies, or contributing factors.

Compare values across time, products, regions, or segments.

Use simple logic or ratios to support your diagnosis.

-If the question is Prescriptive ("What should we do?"):
Suggest 2–3 specific actions to improve or optimize outcomes.

Back each suggestion with data reasoning.

Focus on business impact and next steps.

-If the question is Predictive ("What will happen?"):
Give a rough estimate using average or trend-based projection.

Mention the logic or method used (e.g., last 3 months avg).

Additional Rules:

Be concise: Use bullet points only.

Avoid jargon: Write for business users, not technical teams.

Make it actionable: Every insight should help the business decide what to do next.
"""

    else:
        prompt = f"""
You are a BFSI domain expert and Data Consultant expert.
User asked: \"{user_input}\"
Respond in 4-5 bullet points with useful analysis/suggestions user aked for.
"""
    
    reply = model.generate_content(prompt).text
    st.session_state.chat_history.append({"role": "assistant", "message": reply})
    
    st.rerun()  # Re-render the app to show chat in order
    os.remove("session_token.json")
