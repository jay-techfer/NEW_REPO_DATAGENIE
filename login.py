from flask import Flask, render_template, request, redirect
import pyodbc
import subprocess
import webbrowser
import time 
import threading
import socket
import requests

def get_public_ip():
    return requests.get('https://api.ipify.org').text

public_ip = get_public_ip()
print(public_ip)
app = Flask(__name__)



hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)


# --- DB Function ---
def get_user_credentials():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=EC2AMAZ-89NHHVE\\SQLEXPRESS,50214;'
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
                 "streamlit", "run", "context_based_adventure.py","--server.port","8501","--server.address","0.0.0.0",
                 "--server.headless", "False"
            ])

            # Redirect browser to the Streamlit app
            return redirect(f"http://{public_ip}:8501")
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")

# ✅ THIS PART WAS MISSING!
if __name__ == '__main__':
    def open_browser():
        time.sleep(1)  # Delay to ensure Flask has started
        # webbrowser.open("http://127.0.0.1:5000")
        webbrowser.open(f"http://{public_ip}:5000")
        print(f"DataGenie Running on http://{public_ip}:5000")

    threading.Thread(target=open_browser).start()    
    print("🚀 Flask server is starting...")
    app.run(debug=True, host='0.0.0.0', port=5000) 
    