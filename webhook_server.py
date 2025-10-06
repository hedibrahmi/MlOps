from flask import Flask, request
import subprocess

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    if data and "ref" in data:
        print(f"🚀 Received push on branch {data['ref']}")
        subprocess.Popen(["make", "all"])
        return "Pipeline triggered!", 200
    return "No action", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
