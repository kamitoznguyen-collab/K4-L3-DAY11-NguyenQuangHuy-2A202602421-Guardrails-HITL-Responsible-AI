"""
Lab 11 — Helper Utilities
"""
import asyncio
import time

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner

# Giãn nhịp tối thiểu giữa 2 lượt gọi Gemini để tránh vượt giới hạn request/phút của bản free.
_GEMINI_MIN_INTERVAL = 5.0
_last_gemini_ts = 0.0


async def _pace_gemini() -> None:
    global _last_gemini_ts
    wait = _GEMINI_MIN_INTERVAL - (time.monotonic() - _last_gemini_ts)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_gemini_ts = time.monotonic()

# Bản Gemini free: 429 RESOURCE_EXHAUSTED khi gọi dồn (giới hạn request/phút),
# và 503 UNAVAILABLE khi model quá tải tạm thời. Cả hai đều là lỗi tạm thời →
# tự retry với backoff để CP4 (nhiều lượt gọi liên tiếp) không gãy giữa chừng.
_GEMINI_MAX_RETRIES = 6
_GEMINI_BACKOFF_SECONDS = (6, 12, 24, 40, 60, 60)


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(
        s in msg
        for s in ("429", "resource_exhausted", "exhausted",
                  "503", "unavailable", "high demand", "overloaded",
                  "500", "internal error", "502")
    )


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    last_err: Exception | None = None
    for attempt in range(_GEMINI_MAX_RETRIES):
        final_response = ""
        await _pace_gemini()
        try:
            async for event in runner.run_async(
                user_id=user_id, session_id=session.id, new_message=content
            ):
                if hasattr(event, "content") and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            final_response += part.text
            return final_response, session
        except Exception as e:  # noqa: BLE001
            if not _is_rate_limit_error(e) or attempt == _GEMINI_MAX_RETRIES - 1:
                raise
            wait = _GEMINI_BACKOFF_SECONDS[min(attempt, len(_GEMINI_BACKOFF_SECONDS) - 1)]
            last_err = e
            code = "429" if ("429" in str(e) or "exhaust" in str(e).lower()) else "503"
            print(f"  [gemini {code}] tạm thời — chờ {wait}s rồi thử lại (lần {attempt + 1})…")
            await asyncio.sleep(wait)
    if last_err:
        raise last_err
    return "", session
