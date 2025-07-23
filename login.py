from flask import Flask, render_template, request, redirect
import pyodbc
import subprocess
import webbrowser

app = Flask(__name__)

# --- DB Function ---
def get_user_credentials():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost,1433;'
        'DATABASE=Mahindra;'
        'Trusted_Connection=yes;'
    )
    cursor = conn.cursor()
    cursor.execute("SELECT username, password FROM dbo.personal")
    result = cursor.fetchall()
    return {row[0]: row[1] for row in result}

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        credentials = get_user_credentials()
        username = request.form['username']
        password = request.form['password']

        if username in credentials and credentials[username] == password:
            # Launch Streamlit using `python -m streamlit`
            subprocess.Popen([
                 "streamlit", "run", "context_based_genie.py","--server.port","8501","--server.address","0.0.0.0",
                 "--server.headless", "true"
            ])

            # Redirect browser to the Streamlit app
            return redirect("http://localhost:8501")
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# ✅ THIS PART WAS MISSING!
if __name__ == '__main__':
    print("🚀 Flask server is starting...")
    app.run(debug=True) 
