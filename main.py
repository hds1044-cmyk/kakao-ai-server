from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import re
import asyncio
import httpx
from typing import Any, List, Optional

from openai import OpenAI

app = FastAPI()

OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()

client = (
    OpenAI(
        api_key=OPENAI_API_KEY,
        timeout=httpx.Timeout(12.0, connect=3.0, read=10.0, write=10.0),
        max_retries=0,
    )
    if OPENAI_API_KEY
    else None
)

KAKAO_TEXT_LIMIT = 900
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

    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def extract_image_url(payload: dict) -> Optional[str]:
    utter = (payload.get("userRequest", {}) or {}).get("utterance", "")
    if isinstance(utter, str):
        u = utter.strip()
        if u.startswith(("http://", "https://")):
            return u

    for u in _find_urls_anywhere(payload):
        if IMG_URL_RE.match(u) or "kakaocdn" in u or "kakao" in u:
            return u

    return None


def _safe_user_prompt(user_text: str) -> str:
    t = (user_text or "").strip()
    if not t or t.startswith(("http://", "https://")):
        return "사진 속 영어 문장을 읽고 첨삭해줘."
    return t


def _call_openai_image(image_url: str, user_text: str = "") -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

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
        model="gpt-4o",
        temperature=0.2,
        max_tokens=700,
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


def _call_openai_text(text: str) -> str:
    if not client:
        return "오류: OPENAI_API_KEY가 설정되지 않았습니다."

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


def _kakao_text_response(text: str) -> JSONResponse:
    return JSONResponse(
        {
            "version": "2.0",
            "template": {"outputs": [{"simpleText": {"text": _truncate(text)}}]},
        }
    )


# ✅ 핵심: GET/POST 둘 다 받고, JSON 파싱 실패해도 200으로 카톡에 메시지 뜨게
@app.api_route("/", methods=["POST", "GET"])
async def kakao_skill(request: Request):
    print(f"🔥 HIT {request.method} {request.url.path}")

    # 1) raw body 먼저 읽기(파싱 실패해도 로그 남김)
    raw = await request.body()
    print(f"🔥 RAW LEN {len(raw)}")
    if raw:
        print(f"🔥 RAW HEAD {raw[:300]!r}")

    # 2) JSON 파싱 안전하게
    payload = {}
    if request.method == "POST":
        try:
            payload = await request.json()
        except Exception as e:
            print(f"🔥 JSON PARSE FAIL: {type(e).__name__} {str(e)}")
            return _kakao_text_response("요청(JSON) 파싱에 실패했어요. 카카오 스킬 테스트에서 JSON으로 호출되고 있는지 확인해줘요.")

    # 3) payload 로그(너무 크면 잘라서)
    try:
        import json
        print("===== KAKAO PAYLOAD =====")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:20000])
        print("===== END PAYLOAD =====")
    except Exception:
        pass

    # 4) GET으로 브라우저에서 열었을 때도 확인 가능
    if request.method == "GET":
        return _kakao_text_response("서버는 살아있어요. 카카오 스킬은 POST로 호출해야 정상 동작해요.")

    utter = (payload.get("userRequest", {}) or {}).get("utterance", "") or ""
    img_url = extract_image_url(payload)

    if img_url:
        feedback = await feedback_from_image(img_url, user_text=utter)
        text = f"📷 사진 숙제 피드백\n\n{feedback}"
    else:
        feedback = await feedback_from_text(utter)
        text = f"📘 텍스트 피드백\n\n{feedback}\n\n(사진 URL을 보내면 자동으로 읽고 첨삭해줘요!)"

    return _kakao_text_response(text)
