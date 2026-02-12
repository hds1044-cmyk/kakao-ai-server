from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/")
async def kakao_skill(request: Request):
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "서버 연결 성공 👍"
                    }
                }
            ]
        }
    }
