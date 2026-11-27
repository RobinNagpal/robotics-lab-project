"""Asking Gemini Robotics-ER 2: pictures and a question in, text out.

The API key is read from the environment variable GEMINI_API_KEY, or from the file v3-centrifuge-loading/.env
(copy .env.example to .env and paste the key in; .env is never committed). Google refuses keys without
restrictions (403 Forbidden): in AI Studio's API keys page, restrict the key to the Gemini API.

Every answer is saved in results/raw/ under a name made from the question, the pictures and the settings, so
asking the same thing again costs nothing and the evaluation can be re-scored offline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Pinned: preview models are retired within months (ER 1.5 and 1.6 were).
MODEL = "gemini-robotics-er-2-preview"
API_VERSION = "v1beta"
# A request that has not answered in this long is abandoned and sent again (one once hung for 45 minutes).
TIMEOUT = 120.0
# Google suggests "medium" as the balance between speed and accuracy.
THINKING = "medium"
# Dollars per million tokens until 31 Dec 2026 (then $2 / $10). Thinking is billed as output.
PRICE_IN, PRICE_OUT = 1.0, 5.0

V3 = Path(__file__).resolve().parent.parent
RAW = Path(__file__).resolve().parent / "results" / "raw"


@dataclass
class Answer:
    text: str
    seconds: float
    tokens_in: int
    tokens_out: int  # including thinking
    cached: bool = False
    skipped: bool = False  # no answer within TIMEOUT: left out, not asked again

    @property
    def dollars(self) -> float:
        return (self.tokens_in * PRICE_IN + self.tokens_out * PRICE_OUT) / 1e6


def api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    env = V3 / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GEMINI_API_KEY" and value.strip():
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit(
            f"No Gemini API key: set GEMINI_API_KEY, or copy {V3 / '.env.example'} to .env and fill it in."
        )
    return key


def ask(pngs: list[bytes], prompt: str, thinking: str = THINKING, model: str = MODEL) -> Answer:
    """Send the pictures and the question; the model's text answer (from the saved copy if asked before)."""
    h = hashlib.sha256()
    for part in (model, thinking, prompt, *pngs):
        h.update(part if isinstance(part, bytes) else part.encode())
    path = RAW / f"{h.hexdigest()[:24]}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        return Answer(
            saved["text"],
            saved["seconds"],
            saved["tokens_in"],
            saved["tokens_out"],
            True,
            saved.get("skipped", False),
        )

    from google import genai  # only needed when actually asking
    from google.genai import types

    # The robotics models are previews: listed and served only under the v1beta API version (a 404 otherwise).
    # The HTTP timeout is in milliseconds. It alone was not enough (a request once hung past it), so each
    # request also runs in a thread with a hard deadline: past it, the request is abandoned and sent again.
    client = genai.Client(
        api_key=api_key(),
        http_options=types.HttpOptions(api_version=API_VERSION, timeout=int(TIMEOUT * 1000)),
    )
    content = [
        {"type": "image", "data": base64.b64encode(p).decode(), "mime_type": "image/png"} for p in pngs
    ]
    content.append({"type": "text", "text": prompt})
    for attempt in range(6):
        try:
            t0 = time.perf_counter()
            pool = ThreadPoolExecutor(1)
            job = pool.submit(
                client.interactions.create,
                model=model,
                input=content,
                generation_config={"thinking_level": thinking},
            )
            try:
                reply = job.result(timeout=TIMEOUT + 10)
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
            seconds = time.perf_counter() - t0
            break
        except Exception as e:  # rate limits and server hiccups: wait and try again
            text = str(e)
            if "403" in text:
                raise SystemExit(
                    "403 Forbidden: Google refuses unrestricted keys. In AI Studio's API keys page, restrict "
                    "this key to the Gemini API, then try again."
                ) from e
            retry = (
                "429",
                "RESOURCE_EXHAUSTED",
                "500",
                "503",
                "UNAVAILABLE",
                "timed out",
                "Timeout",
                "timeout",
            )
            if "RESOURCE_EXHAUSTED" in text or "exceeded your current quota" in text:
                raise SystemExit(
                    "Gemini quota used up (429): the key has hit its limit. Wait for it to reset, or enable "
                    "billing on its Google project. Answers so far are saved; re-running asks only the rest."
                ) from e
            if "Timeout" in type(e).__name__ or "timed out" in text:
                # A question Gemini does not answer in time is left out, not asked again: retrying cost time
                # and money for nothing. Saved as skipped, so a re-run does not ask it either.
                print(f"  Gemini: no answer in {TIMEOUT:.0f} s, question skipped", flush=True)
                RAW.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps(
                        {
                            "model": model,
                            "thinking": thinking,
                            "prompt": prompt,
                            "text": "",
                            "seconds": TIMEOUT,
                            "tokens_in": 0,
                            "tokens_out": 0,
                            "skipped": True,
                        }
                    )
                )
                return Answer("", TIMEOUT, 0, 0, skipped=True)
            if attempt == 5 or not any(s in text for s in retry):
                raise
            print(f"  Gemini: {type(e).__name__} {text[:80]}, trying again ({attempt + 1} of 5)", flush=True)
            time.sleep(10 * 2**attempt)
    usage = reply.usage
    tokens_in = int(getattr(usage, "total_input_tokens", 0) or 0)
    tokens_out = int(getattr(usage, "total_output_tokens", 0) or 0) + int(
        getattr(usage, "total_thought_tokens", 0) or 0
    )
    answer = Answer(reply.output_text or "", seconds, tokens_in, tokens_out)
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": model,
                "thinking": thinking,
                "prompt": prompt,
                "text": answer.text,
                "seconds": seconds,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            },
            indent=1,
        )
    )
    return answer
