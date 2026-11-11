from fastapi import FastAPI

app = FastAPI(title="Invoice-to-Accounting API")

@app.get("/health")
def health():
    return {"status": "ok"}