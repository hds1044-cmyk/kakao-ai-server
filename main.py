from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import re
import asyncio
import httpx
from typing import Any, List, Optional

from openai import OpenAI

app = FastAPI()

# ===== OpenAI Client (정확도 우선 + 카톡 스킬오류 방지용 설정) =====
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

client = (
    OpenAI(
        api_key=OPENAI_API_KEY,
        # 카카오 응답 시간 제한을 넘기지 않게, 연결/읽기 타임아웃을 과도하게 길게 두지 않음
        timeout=httpx.Timeout(12.0, connect=3.0, read=10.0, write=10.0),
        # retry가 붙으면 응답이 더 늦어져 카톡에서 스킬오류(타임아웃) 확률이 올라감
        max_retries=0,
    )
    if OPENAI_API_KEY
    else None
)

# ===== 카카오 말풍선 길이 제한 대비 =====
KAKAO_TEXT_LIMIT = 900

# 카카오 스킬 응답 제한 대비 "하드 타임아웃"
# (환경에 따라 5초/10초 등 다를 수 있어서, 일단 9초로 두고 문제가 있으면 4.5초로 낮추세요)
KAKAO_HARD_TIMEOUT_SEC = 4.5

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


def _safe_user_prompt(user_text: str) -> str:
    t = (user_text or "").strip()
    # 사용자가 URL만 보낸 경우 프롬프트가 URL이 되지 않게 처리
    if not t or t.startswith(("http://", "https://")):
        return "사진 속 영어 문장을 읽고 첨삭해줘."
    return t


def _call_openai_image(image_url: str, user_text: str = "") -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

    # 정확도 우선 프롬프트
    system = (
        "너는 영어 숙제 첨삭 선생님이야. "
        "사진 속 영어 문장을 읽고 한국어로 피드백해. "
        "틀린/어색한 문장만 골라서 항목별로 번호를 붙여서 답해.\n"
        "각 항목은 아래 순서로:\n"
        "수정문장\n"
        "이유\n"
        "너무 장문으로 쓰지 마."
    )

    prompt_text = _safe_user_prompt(user_text)

    resp = client.chat.completions.create(
        model="gpt-4o",          # ✅ 정확도 우선
        temperature=0.2,
        max_tokens=700,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},  # ✅ high
                ],
            },
        ],
    )
    return _truncate(resp.choices[0].message.content or "")


def _call_openai_text(text: str) -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

    # 텍스트도 정확도 우선으로 gpt-4o 사용
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        max_tokens=500,
        messages=[
            {"role": "system", "content": "너는 영어 첨삭 선생님이야. 틀린/어색한 부분만 간단히 한국어로 고쳐줘."},
            {"role": "user", "content": (text or "").strip()},
        ],
    )
    return _truncate(resp.choices[0].message.content or "")


async def feedback_from_image(image_url: str, user_text: str = "") -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_openai_image, image_url, user_text),
            timeout=KAKAO_HARD_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        # 카카오 스킬오류 방지: 무조건 응답을 돌려주기
        return (
            "사진 판독에 시간이 조금 더 걸리고 있어요. "
            "문장을 텍스트로 보내주시면 정확하게 바로 첨삭해드릴게요."
        )
    except Exception as e:
        return _truncate(f"오류: {type(e).__name__} | {str(e)}")


async def feedback_from_text(text: str) -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_openai_text, text),
            timeout=KAKAO_HARD_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        return (
            "응답이 지연되고 있어요. "
            "문장을 조금 나눠서 보내주시면 더 정확하고 안정적으로 첨삭돼요."
        )
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
        text = f"📘 텍스트 피드백\n\n{feedback}\n\n(사진 URL을 보내면 자동으로 읽고 첨삭해줘요!)"

    return JSONResponse(
        {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": text}}]},
        }
    )
