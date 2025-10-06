from fastapi import FastAPI, Request
import subprocess

app = FastAPI(title="MLOps Webhook Server")

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    if data and "ref" in data:
        print(f"🚀 Received push on branch {data['ref']}")
        subprocess.Popen(["make", "all"])
        return {"message": "Pipeline triggered!"}
    return {"message": "No action"}

