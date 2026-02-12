from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os

from openai import OpenAI

app = FastAPI()

# ✅ Cloud Run 환경변수에서 키 읽기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/")
async def kakao_skill(request: Request):
    body = await request.json()

    # ✅ 카카오 i 오픈빌더: 사용자 발화는 여기 들어옴
    user_text = body.get("userRequest", {}).get("utterance", "")

    # 키가 없으면 바로 안내
    if not client:
        feedback = "오류: OPENAI_API_KEY가 설정되지 않았습니다."
    else:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an English teacher. Give short and clear feedback."},
                    {"role": "user", "content": user_text},
                ],
            )
            feedback = response.choices[0].message.content or "(빈 응답)"
        except Exception as e:
            feedback = f"오류: {type(e).__name__} | {str(e)}"

    # ✅ 카카오 응답 포맷
    return JSONResponse(
        {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": f"📒 AI 피드백:\n\n{feedback}"
                        }
                    }
                ]
            },
        }
    )
