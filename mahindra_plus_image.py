import socket
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import urllib
import re
import pyodbc
import numpy as np
from sqlalchemy import create_engine
import google.generativeai as genai
import traceback
import base64
import io
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import json


# Load the image from local file
logo_image = Image.open("techfer_logo_new.png")  # replace with your actual logo file name

# Set page config with image as icon
st.set_page_config(
    page_title="QueryGenie",
    page_icon=logo_image,  # 🖼️ your custom image here
    layout="centered",     # 👈 Change this from "wide" to "centered"
    initial_sidebar_state="expanded"
)


# Add logo and title


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


with open("bgg.jpg", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode()

base64_img = get_base64_image("techfer_logo_new.png")

st.markdown(
    f"""
    <style>
    /* Set background image */
    body {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
    }}

    .stApp {{
        background-color: rgba(255, 255, 255, 0.01);
    }}

    /* Sidebar & transparent elements */
    header, .stDeployButton, .st-emotion-cache-6qob1r, .st-emotion-cache-1avcm0n {{
        background-color: transparent !important;
    }}

    .stTextArea, .stTextInput, .st-emotion-cache-1wmy9hl {{
        background-color: transparent !important;
    }}
    textarea {{
    background-color: rgba(0, 0, 0, 0.01) !important;
    color: black !important;
    border: 1px solid white !important;
    border-radius: 8px !important;
    padding: 0.5em 1.2em !important;
    font-weight: bold;
    box-shadow: 0 0 8px rgba(0,0,0,0.2);
    min-height: 60px !important;
    # height: auto !important;
    # overflow-y: hidden !important;
    # resize: none !important;
    }}
    label {{
        color: Black !important;
        font-weight: bold;
        font-size: 16px;
    }}

    .stButton>button {{
        background-color: rgba(0, 0, 0, 0.2) !important;
        color: black !important;
        border: 1px solid white !important;
        border-radius: 8px !important;
        padding: 0.5em 1.2em !important;
        font-weight: bold;
        box-shadow: 0 0 8px rgba(0,0,0,0.2);
    }}

    .stButton>button:hover {{
        background-color: rgba(255, 255, 255, 0) !important;
        transition: 0.3s ease;
    }}

    /* Title layout */
    .title-container {{
        display: flex;
        flex-direction: column;
        align-items: left;
        gap: 2px;
        padding: 1px;
        border-radius: 10px;
        background-color: transparent;
    }}

    .main-title {{
        font-size: 48px;
        color: black;
        font-weight: 700;
        margin: -10;
        text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.1);
    }}

    /* Fixed top-left logo */
    .fixed-logo {{
        position: fixed;
        top: 50px;
        left: 70px;
        height: 60px;
        width: auto;
        z-index: 1000;
    }}
    </style>

    <script>
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {{
        textarea.setAttribute('style', 'height:auto;overflow-y:hidden;');
        textarea.addEventListener('input', function() {{
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        }}, false);
    }});
    </script>

    <!-- Fixed logo -->
    <img src="data:image/jpeg;base64,{base64_img}" class="fixed-logo">

    <!-- Title -->
    <div class="title-container">
        <div class="main-title">QueryGenie<br><h5>🧠 SQL Query Generator & Executor</h5></div>
    </div>
    """,
    unsafe_allow_html=True
)
# --------------------- DATABASE ---------------------

# Gemini API configuration
genai.configure(api_key="AIzaSyDtdF-8PJ0GiQUXiN9oyO387V2lTL8tr3g")
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={
        "temperature": 0.2,
        "top_p": 1.0,
        "top_k": 40,
        "max_output_tokens": 2048
    }
)

# SSMS Server details

# Automatically get local IP address
# hostname = socket.gethostname()
# local_ip = socket.gethostbyname(hostname)

ssms_servers = [
    {
        "name": "VIJAY\\SQLEXPRESS",   # or just "SQLEXPRESS"
        "server": "VIJAY\SQLEXPRESS,52235",         # dynamically set IP
        "username": "sa",
        "password": "abcd123456"
    }
]


# Global DataFrames
ssms_schema_df = pd.DataFrame()


@st.cache_data(show_spinner=False)
def fetch_ssms_schema():
    data = []
    for s in ssms_servers:
        try:
            # Directly connect to Mahindra database only
            conn = f"Driver={{SQL Server}};Server={s['server']};UID={s['username']};PWD={s['password']};Database=Mahindra;Encrypt=no;"
            engine = create_engine(f"mssql+pyodbc:///?odbc_connect={urllib.parse.quote_plus(conn)}")
            df = pd.read_sql(
                "SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS",
                engine
            )
            df['SERVER'] = s['name']
            df['DATABASE'] = "Mahindra"
            data.append(df)
        except Exception as e:
            st.warning(f"SSMS Error: {e}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()



def gen_join_queries(user_input):
    user_input_add = user_input.strip().rstrip('.') 
    schema_description = ""
    if os.path.exists("parameters.json"):
                        with open("parameters.json", "r", encoding="utf-8") as f:
                            param_info = json.load(f)
                            schema_description += "[parameters.json]\n"
                            schema_description += "\n".join([
                                f"- {item['Parameter']}: {item.get('Description', item.get('Descripsition', 'No description'))}"
                                for item in param_info
                            ])
                            schema_description += "\n"
    else:
        schema_description += "[parameters.json]\nSchema details not available.\n"
                
                    # Handle the other flat dictionary JSON files
    flat_json_files = ["parts.json", "labour.json", "verbatim.json"]

    for file in flat_json_files:
        if os.path.exists(file):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                schema_description += f"[{file}]\n"
                for key, value in data.items():
                    schema_description += f"- {key}: {value}\n"
        else:
            schema_description += f"[{file}]\nSchema details not available.\n"

                    # Prompt for model
        prompt = f"""
You are a helpful assistant that generates valid and optimized T-SQL queries for SQL Server.
The user is working with tables called `dbo.sampledata`, `dbo.parts`, `dbo.labour`, and `dbo.verbatim` inside the `Mahindra` database.

Schema details:
{schema_description}

Strict Instructions:
- Do NOT use CTEs (e.g., WITH ... AS) or `ROW_NUMBER()` functions.
- Do NOT use `TOP` in the `SELECT` clause **unless the user specifically requests a limit** (e.g., "top 5", "top 10", etc.).
- Always use `SELECT TOP XX` format instead of `ROW_NUMBER()` for limiting rows.
- Use `COUNT(*) AS Repetition_Count` when showing how many times something occurred.
- For analyzing most frequent entries, use `GROUP BY` followed by `ORDER BY COUNT(*) DESC`.
- Use clear aliases like `Source_Type` and `Item_Description` when combining records from different tables.
- For date filtering, use: `RO_Date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'`.
- Do not wrap multiple GROUP BYs inside subqueries unless required — keep it simple and flat.
- Output only the query and nothing else.
- while generating the query use [Mahindra].[dbo].[table_name]

example:
-if user ask "How much discount in part & labour was given in MH47AG8538 ,KA03NE7738 and UP16EB8849"
give query "SELECT
	[Mahindra].[dbo].[sampledata].[PART_DISCNT_PERCNTG]  AS Total_Part_Discount_Percentage,
	[Mahindra].[dbo].[sampledata].[LABR_DISCNT_PERCNTG]  AS Total_Labour_Discount_Percentage
FROM
	[Mahindra].[dbo].[sampledata]
WHERE
	[Mahindra].[dbo].[sampledata].[REG_NUMBR] IN ('MH47AG8538', 'KA03NE7738', 'UP16EB8849')"
don't put sum when not required



User Question:
{user_input_add}
"""

    response = model.generate_content(prompt)

    sql_query = response.text.replace("```sql", "").replace("```", "").strip()
    return sql_query

def generate_chart_code(df):
    sample_data = df.columns
    prompt = f"""
You are a Python data visualization assistant.

Based on this data sample:

{sample_data}

Generate a Python Plotly code block that creates one meaningful chart from this dataset.
Use appropriate labels and titles.
Use `fig = ...` to define the Plotly figure (required).
Output only valid Python code (no comments, no explanation).
You can use Plotly Express (px) or Graph Objects (go).
You can use NumPy for array arithmetic.
    """
    response = model.generate_content(prompt).text
    code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
    return code_match.group(1).strip() if code_match else ""


def render_chart(code, df):
    exec_globals = {
        "pd": pd,
        "df": df,
        "px": px,
        "go": go,
        "np": np
    }

    try:
        exec(code, exec_globals)
        fig = exec_globals.get("fig", None)
        return fig if fig else None
    except Exception as e:
        st.error("Chart generation failed.")
        st.exception(e)
        return None


# Load schema once
if ssms_schema_df.empty:
    with st.spinner("Loading all dependencies..."):
        ssms_schema_df = fetch_ssms_schema()
    st.success("✅ Ready to Use")

# User input
user_input = st.text_input("Ask a question based on the database schema:")
submit = st.button("Generate SQL & Chart")

try:
    if submit and user_input:
        with st.spinner("Generating SQL..."):

                ssms_schema_txt = ssms_schema_df.to_string(index=False).strip()
                ssms_schema_txt = re.sub(r'\s{2,}', ' ', ssms_schema_txt)

                sql_text = gen_join_queries(user_input)
                print('sql query: ',sql_text)
                st.subheader("Generated SQL Query")

                # query = sql_text.group(1).strip()
                # query = re.sub(r"^```sql", "", query,
                #                 flags=re.IGNORECASE).strip()
                query = sql_text.strip("`").rstrip(";").strip()
                print('query',query)

                st.session_state["last_query"] = query  # ✅ Store query

                st.code(query, language="sql")

                server_cfg = ssms_servers[0]
                conn_str = (
                    f"Driver={{ODBC Driver 17 for SQL Server}};"
                    f"Server={server_cfg['server']};"
                    f"UID={server_cfg['username']};"
                    f"PWD={server_cfg['password']};"
                    f"Encrypt=no;"
                    f"TrustServerCertificate=yes;"
                )

                quoted_conn = urllib.parse.quote_plus(conn_str)
                engine = create_engine(
                    f"mssql+pyodbc:///?odbc_connect={quoted_conn}")
                new_df1 = pd.read_sql(query, engine)

                # ✅ Save DF for later popup
                st.session_state["query_result_df"] = new_df1
                # ✅ Reset popup state
                st.session_state["show_popup"] = False
                st.success(
                    "✅ Query executed. You can now open the chart generator.")
    else:
        print('No valid query found in response.')

except Exception as e:
    st.error(f"An error occurred. {e}")
    st.warning("No valid query found in response.")
    st.exception(traceback.format_exc())

# Show query result (if already stored)
if "query_result_df" in st.session_state:
    new_df1 = st.session_state["query_result_df"]
    st.subheader("SSMS Query Result")
    st.dataframe(new_df1, use_container_width=True)

    # Clear previous accepted charts and generated chart if data changes
    if "last_df_shape" not in st.session_state or st.session_state["last_df_shape"] != new_df1.shape:
        st.session_state.accepted_charts = []
        if "generated_chart_code" in st.session_state:
            del st.session_state["generated_chart_code"]
        st.session_state["last_df_shape"] = new_df1.shape

    # Section for accepted charts (store chart code instead of figures)
    if "accepted_charts" not in st.session_state:
        st.session_state.accepted_charts = []

    if st.button("📊 Open Chart Generator"):
        st.session_state["show_popup"] = True

    if st.session_state.get("show_popup", False):
        with st.expander("🛠️ Chart Generator", expanded=True):
            left_col, right_col = st.columns([1, 2])

            with left_col:
                st.markdown("### Select Columns")
                selected_cols = []
                for col in new_df1.columns:
                    if st.checkbox(col, key=f"col_{col}"):
                        selected_cols.append(col)

            with right_col:
                st.markdown("### Customize Your Chart")
                chart_prompt = st.text_area(
                    "📝 What kind of chart do you want to generate?", height=100, key="chart_desc")

                if st.button("🎨 Generate Chart"):
                    try:
                        if not chart_prompt or not selected_cols:
                            st.warning(
                                "Please enter chart description and select columns.")
                        else:
                            column_list = ", ".join(selected_cols)
                            custom_prompt = f"""
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

                            response = model.generate_content(
                                custom_prompt).text
                            print(response)
                            chart_code = re.search(
                                r"```python(.*?)```", response, re.DOTALL)
                            code_clean = chart_code.group(
                                1).strip() if chart_code else ""

                            st.session_state["generated_chart_code"] = code_clean

                    except Exception as e:
                        st.error("Chart generation failed.")
                        st.exception(e)

    # Show generated chart immediately and offer Accept/Reject
    if "generated_chart_code" in st.session_state:
        exec_globals = {
            "pd": pd,
            "df": new_df1,
            "px": px,
            "go": go,
            "np": np
        }
        try:
            exec(st.session_state["generated_chart_code"], exec_globals)
            for var_name in exec_globals:
                if re.match(r"fig\d*$", var_name) and isinstance(exec_globals[var_name], go.Figure):
                    fig = exec_globals[var_name]
                    st.plotly_chart(fig, use_container_width=True,
                                    key=f"preview_{var_name}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"✅ Accept {var_name}", key=f"accept_{var_name}"):
                            if len(st.session_state.accepted_charts) < 5:
                                st.session_state.accepted_charts.append(
                                    st.session_state["generated_chart_code"])
                                del st.session_state["generated_chart_code"]
                                st.rerun()

                            else:
                                st.warning(
                                    "Maximum of 4 charts accepted. Please remove one before adding more.")
                    with col2:
                        if st.button(f"❌ Reject {var_name}", key=f"reject_{var_name}"):
                            del st.session_state["generated_chart_code"]
                            st.rerun()

        except Exception as e:
            st.error("Chart display failed.")
            st.exception(e)

    # Display all accepted charts in dynamic grid with delete option
    if st.session_state.accepted_charts:
        st.markdown("---")
        st.markdown("## 📊 Accepted Visualizations")

        # Display charts in 2 columns per row
        for i in range(0, len(st.session_state.accepted_charts), 2):
            row = st.columns(2)
            for j in range(2):
                idx = i + j
                if idx >= len(st.session_state.accepted_charts):
                    break

                chart_code = st.session_state.accepted_charts[idx]
                exec_globals = {
                    "pd": pd,
                    "df": new_df1,
                    "px": px,
                    "go": go,
                    "np": np
                }
                try:
                    exec(chart_code, exec_globals)
                    for var_name in exec_globals:
                        if re.match(r"fig\d*$", var_name) and isinstance(exec_globals[var_name], go.Figure):
                            fig = exec_globals[var_name]
                            with row[j]:
                                st.plotly_chart(fig, use_container_width=True,
                                                key=f"accepted_{idx}_{var_name}")
                                if st.button(f"🗑️ Delete {var_name}", key=f"delete_{idx}_{var_name}"):
                                    st.session_state.accepted_charts.pop(idx)
                                    st.rerun()
                except Exception as e:
                    st.error(f"Chart {idx+1} rendering failed.")
                    st.exception(e)
