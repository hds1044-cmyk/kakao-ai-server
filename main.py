from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import re
from typing import Any, List, Optional

from openai import OpenAI

app = FastAPI()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# 카카오 말풍선 길이 제한 대비
KAKAO_TEXT_LIMIT = 900

IMG_URL_RE = re.compile(r"^https?://.+\.(png|jpg|jpeg|webp)(\?.*)?$", re.IGNORECASE)


@app.get("/health")
def health():
    return {"ok": True}


def _truncate(text: str, limit: int = KAKAO_TEXT_LIMIT) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 10].rstrip() + "\n...(이하 생략)"


def _find_urls_anywhere(obj: Any) -> List[str]:
    urls: List[str] = []

    def walk(x: Any):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            s = x.strip()
            if s.startswith("http://") or s.startswith("https://"):
                urls.append(s)

    walk(obj)
    # 중복 제거(순서 유지)
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def extract_image_url(payload: dict) -> Optional[str]:
    # 1) 가장 흔함: userRequest.utterance 에 이미지 URL이 그대로 들어옴
    utter = (payload.get("userRequest", {}) or {}).get("utterance", "")
    if isinstance(utter, str):
        u = utter.strip()
        if u.startswith(("http://", "https://")):
            return u

    # 2) 혹시 다른 필드로 들어오는 경우 전체 스캔
    for u in _find_urls_anywhere(payload):
        if IMG_URL_RE.match(u) or "kakaocdn" in u or "kakao" in u:
            return u

    return None


async def feedback_from_image(image_url: str, user_text: str = "") -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

    # 사진 숙제 첨삭 프롬프트(원하는 스타일로 나중에 더 맞춰줄게)
    system = (
        "너는 영어 숙제 첨삭 선생님이야. "
        "사진 속 영어 문장을 읽고 아래 형식으로 한국어로 답해:\n"
        "1) 틀린/어색한 부분\n"
        "2) 수정본(자연스럽게)\n"
        "3) 짧은 이유/팁\n"
        "가능하면 항목별로 번호를 붙이고, 너무 길게 쓰지 마."
    )

    # 사용자가 사진과 함께 한마디를 적으면 같이 반영
    prompt_text = (user_text or "").strip() or "이 사진 숙제를 읽고 첨삭해줘."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
                    ],
                },
            ],
        )
        return _truncate(resp.choices[0].message.content or "")
    except Exception as e:
        return _truncate(f"오류: {type(e).__name__} | {str(e)}")


async def feedback_from_text(text: str) -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "너는 영어 첨삭 선생님이야. 짧고 명확하게 한국어로 피드백해."},
                {"role": "user", "content": text or ""},
            ],
        )
        return _truncate(resp.choices[0].message.content or "")
    except Exception as e:
        return _truncate(f"오류: {type(e).__name__} | {str(e)}")


@app.post("/")
async def kakao_skill(request: Request):
    payload = await request.json()

    utter = (payload.get("userRequest", {}) or {}).get("utterance", "") or ""
    img_url = extract_image_url(payload)

    if img_url:
        feedback = await feedback_from_image(img_url, user_text=utter)
        text = f"📷 사진 숙제 피드백\n\n{feedback}"
    else:
        feedback = await feedback_from_text(utter)
        text = f"📘 텍스트 피드백\n\n{feedback}\n\n(사진으로 보내면 자동으로 읽고 첨삭해줘요!)"

    return JSONResponse(
        {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": text}}]},
        }
    )
