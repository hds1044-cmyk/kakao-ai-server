from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/")
async def kakao_skill(request: Request):
    body = await request.json()
    user_text = body.get("userRequest", {}).get("utterance", "")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an English teacher. Give short and clear feedback."},
                {"role": "user", "content": user_text}
            ]
        )

        feedback = response.choices[0].message.content

    except Exception as e:
        feedback = "AI 응답 중 오류가 발생했습니다."

    return JSONResponse({
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": f"📘 AI 피드백:\n\n{feedback}"
                    }
                }
            ]
        }
    })
