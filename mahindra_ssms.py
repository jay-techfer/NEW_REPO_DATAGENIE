import streamlit as st
import pandas as pd
import json
import base64
import pyodbc
from PIL import Image
import google.generativeai as genai
import traceback
import os
import traceback


# --------------------- CONFIG & STYLE ---------------------
# Load images
logo_image = Image.open("techfer_logo_new.png")
bg_image_path = "bgg.jpg"

# Page setup
st.set_page_config(
    page_title="QueryGenie",
    page_icon=logo_image,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Encode images to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

base64_img = get_base64_image("techfer_logo_new.png")
img_base64 = get_base64_image(bg_image_path)

# Inject CSS & HTML
st.markdown(
    f"""
    <style>
    body {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center center;
        background-repeat: no-repeat;
    }}
    .stApp {{
        background-color: rgba(255, 255, 255, 0.01);
    }}
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
    }}
    label {{
        color: black !important;
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
    .fixed-logo {{
        position: fixed;
        top: 50px;
        left: 70px;
        height: 60px;
        width: auto;
        z-index: 1000;
    }}
    </style>

    <img src="data:image/png;base64,{base64_img}" class="fixed-logo">

    <div class="title-container">
        <div class="main-title">QueryGenie<br><h5>🧠 SQL Query Generator & Executor</h5></div>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------- GEMINI API ---------------------
genai.configure(api_key="AIzaSyCZcqg-ww8nEazLajnW6dJoVxxdl9zye5M")  # Replace with your actual key
model = genai.GenerativeModel("gemini-1.5-flash")

# --------------------- DATABASE ---------------------
def get_db_connection():
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=VIJAY\SQLEXPRESS,52235;"
        "DATABASE=Mahindra;"
        "UID=sa;"
        "PWD=abcd123456"
    )
    return pyodbc.connect(conn_str)

# --------------------- MAIN UI ---------------------
user_question = st.text_area("Ask me anything", height=100)

if st.button("Generate & Execute Query"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Thinking..."):
                schema_description = ""

                # Handle parameters.json (list of dicts with Parameter + Description)
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
You are a helpful assistant that generates T-SQL queries for SQL Server.
The user is working with tables called `dbo.sampledata`, `dbo.parts`, `dbo.labour`, and `dbo.verbatim` inside the `Mahindra` database.

Here is the table schema with column descriptions to help you understand the context:
{schema_description}
Important Notes:
- Use `DATEDIFF` between `RO_DATE` and `CLOSD_DATE` to calculate delivery duration.
- In the `parts` table:
  - `MODEL_GROP` and `FAMLY_DESC` are **two separate fields**. Do NOT treat them as interchangeable.
  - `Dealer_Zone`, `Dealer_Area_Office`, `Dealer_Parent_Name`, and `Dealer_Location` are all **distinct** fields. Match them exactly.
  - Use `CAST([Part_Quantity] AS INT)` and `CAST([part_basic_amount] AS FLOAT)` for arithmetic operations.
  - Group by both `PART_DESC` and `PART_NUMBR` when analyzing part replacements.
  - Always ORDER BY `TotalQuantity DESC, TotalValue DESC` when showing frequently replaced parts.
  -For text-based filters, match entire strings exactly as they appear in the column — do not simplify or shorten them
Now, based on the following natural language question, generate a valid and optimized SQL query **only** (do not explain):



Question: {user_question}
"""

                response = model.generate_content(prompt)

                sql_query = response.text.replace("```sql", "").replace("```", "").strip()
                st.code(sql_query, language="sql")

                # Execute SQL
                conn = get_db_connection()
                df = pd.read_sql(sql_query, conn)
                conn.close()

                st.success("Query executed successfully!")
                st.dataframe(df)

        except Exception as e:
            st.error("❌ Error occurred while processing your request.")
            st.exception(e)
            st.text(traceback.format_exc())
