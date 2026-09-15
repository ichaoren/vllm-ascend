#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Replay dumped requests over HTTP.

Input is the request dump written by ``load_balance_proxy_server_example.py``
when ``VLLM_DUMP_REQ_PATH`` is set: ``<dir>/requests_<pid>.jsonl``, one json
object per line with ``req_id``, ``api`` and ``body``. Each body is POSTed to
the same endpoint it originally hit, so the replay goes through the identical
serving path.

The replay reproduces the **original proxy entry request**: nothing is added. No
``X-Request-Id`` header is sent (the proxy mints its own id, as it does for a
real client) and the client's original ``stream`` setting is kept.

Two groups of fields are dropped, both written by the proxy rather than the
client:

* ``kv_transfer_params`` — written back from the prefiller's response before the
  dump is taken; its KV coordinates (``remote_block_ids``, ``remote_engine_id``,
  ``remote_host`` / ``remote_port``) are one-shot and point at blocks freed long
  ago. The proxy fills them in again on its own.
* ``max_tokens`` / ``min_tokens`` / ``max_completion_tokens`` — the prefill leg
  pins the generation length to 1 and dumps carry that value. Replaying it would
  stop after a single token and never reach the decode steps. Pass
  ``--max-tokens N`` for an explicit budget instead of the server default.

The dumped ``api`` field is backend-relative (``/chat/completions``) because the
proxy hands that form to its backend clients, whose base_url already ends in
``/v1``. It is rewritten to the proxy's public route (``/v1/chat/completions``)
before replaying; posting the dumped path verbatim would 404.

Point ``--base-url`` at the **proxy**, not at a decode node: a node started with
``kv_role: kv_consumer`` cannot prefill on its own.

Prefix caching (``--enable-prefix-caching``) makes a replayed prompt hit the
cache and take a different path than the original request. Two opt-in ways to
deal with that: ``--reset-prefix-cache`` flushes the cache first and keeps the
bodies untouched, or ``--cache-salt`` injects a unique salt per request (which
does modify the body).

Usage::

    python examples/disaggregated_prefill_v1/replay_spec_reject.py \\
        --input '/path/to/dump/requests_*.jsonl' \\
        --base-url http://127.0.0.1:9000 \\
        --max-tokens 256 --repeat 3

``--input`` accepts glob patterns, so ``'requests_*.jsonl'`` picks up every
uvicorn worker's dump. Use ``--limit`` to replay only the first N requests.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import sys
import uuid
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover
    print("httpx is required: pip install httpx", file=sys.stderr)
    raise


def load_records(pattern: str) -> list[dict[str, Any]]:
    """Load dump records from one file or a glob of per-worker dumps."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        # Not a glob, or nothing matched: fall back to the literal path so the
        # caller sees a normal "file not found" instead of an empty replay.
        paths = [pattern]
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"warning: {path}:{lineno} bad json: {e}", file=sys.stderr)
                    continue
                if rec.get("body") is None:
                    print(f"warning: {path}:{lineno} has no body, skipping", file=sys.stderr)
                    continue
                records.append(rec)
    return records


def resolve_endpoint(rec: dict[str, Any]) -> str:
    """Map a dumped ``api`` value onto the proxy's public route.

    The dump stores the value the proxy passes to ``handle_completions_impl``,
    which is backend-relative (``/chat/completions``, ``/completions``) because
    the backend clients' base_url already ends in ``/v1``. The proxy's own routes
    are ``/v1/chat/completions`` and ``/v1/completions``, so replaying the dumped
    path verbatim gets a 404. Prefix ``/v1`` when it is missing.

    If the value is unusable, fall back to the request body's shape: ``messages``
    means chat, ``prompt`` means completions.
    """
    api = (rec.get("api") or "").strip()
    if api and not api.startswith("/"):
        api = "/" + api
    if api in ("/v1/chat/completions", "/v1/completions"):
        return api
    if api in ("/chat/completions", "/completions"):
        return "/v1" + api
    body = rec.get("body") or {}
    inferred = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    print(
        f"warning: req {rec.get('req_id')} has unrecognised api {api!r}; "
        f"using {inferred} inferred from the body",
        file=sys.stderr,
    )
    return inferred


_PROXY_INJECTED_FIELDS = (
    "kv_transfer_params",
    # `build_prefill_request` pins the generation length to 1 for the prefill
    # leg. Dumps observed in the wild carry `max_tokens: 1`, which would cap the
    # replay at a single token and never reach the decode steps at all, so the
    # length fields are dropped and the server default applies. Use
    # --max-tokens to set an explicit budget.
    "max_tokens",
    "min_tokens",
    "max_completion_tokens",
)


def prepare_body(rec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    body = dict(rec["body"])
    for field in _PROXY_INJECTED_FIELDS:
        body.pop(field, None)
    if args.max_tokens > 0:
        # Chat and completions both accept max_tokens; max_completion_tokens is
        # the newer chat-only spelling and is left out on purpose so one flag
        # covers both endpoints.
        body["max_tokens"] = args.max_tokens
    if args.cache_salt:
        # Opt-in: a distinct salt makes the prefix cache treat this as a fresh
        # prompt, at the cost of no longer being byte-identical to the original.
        body["cache_salt"] = f"replay-{uuid.uuid4()}"
    if args.no_stream:
        # Opt-in: forcing non-streaming simplifies the client side. It does not
        # change how the engine drafts or verifies tokens.
        body["stream"] = False
        body.pop("stream_options", None)
    return body


async def replay_one(
    client: httpx.AsyncClient,
    rec: dict[str, Any],
    attempt: int,
    args: argparse.Namespace,
    sem: asyncio.Semaphore,
) -> dict[str, Any]:
    api = resolve_endpoint(rec)
    body = prepare_body(rec, args)
    # No headers at all: the proxy mints its own request id, exactly as it does
    # for a real client. Passing X-Request-Id here would override it.
    async with sem:
        try:
            if body.get("stream"):
                return await _replay_streaming(client, api, body, rec, attempt, args)
            resp = await client.post(api, json=body, timeout=args.timeout)
        except Exception as e:  # noqa: BLE001 - report and continue
            return {"req_id": rec.get("req_id"), "attempt": attempt, "error": str(e)}
    result: dict[str, Any] = {
        "req_id": rec.get("req_id"),
        "attempt": attempt,
        "status": resp.status_code,
    }
    if resp.status_code != 200:
        result["error"] = resp.text[:500]
        return result
    try:
        payload = resp.json()
        usage = payload.get("usage") or {}
        result["completion_tokens"] = usage.get("completion_tokens")
        result["new_req_id"] = payload.get("id")
        result["text"] = _extract_text(payload)
    except Exception:  # noqa: BLE001 - unexpected non-json body
        result["completion_tokens"] = None
    return result


def _extract_text(payload: dict[str, Any]) -> str:
    """Pull the generated text out of a non-streaming response.

    Covers both shapes: chat puts it in ``choices[].message.content``, plain
    completions in ``choices[].text``. Reasoning models may also return
    ``reasoning_content``, which is included so a garbled reasoning trace is
    visible too.
    """
    out = []
    for choice in payload.get("choices") or []:
        message = choice.get("message") or {}
        for value in (message.get("reasoning_content"), message.get("content"), choice.get("text")):
            if value:
                out.append(value)
    return "".join(out)


async def _replay_streaming(
    client: httpx.AsyncClient,
    api: str,
    body: dict[str, Any],
    rec: dict[str, Any],
    attempt: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[str, Any] = {"req_id": rec.get("req_id"), "attempt": attempt, "streamed": True}
    chunks = 0
    text_parts: list[str] = []
    async with client.stream("POST", api, json=body, timeout=args.timeout) as resp:
        result["status"] = resp.status_code
        if resp.status_code != 200:
            await resp.aread()
            result["error"] = resp.text[:500]
            return result
        async for line in resp.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:") :].strip()
            if not payload_str or payload_str == "[DONE]":
                continue
            chunks += 1
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            if result.get("new_req_id") is None:
                result["new_req_id"] = payload.get("id")
            usage = payload.get("usage")
            if usage:
                result["completion_tokens"] = usage.get("completion_tokens")
            # Accumulate the incremental text. Chat streams carry it in
            # choices[].delta, plain completions in choices[].text.
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}
                for value in (delta.get("reasoning_content"), delta.get("content"), choice.get("text")):
                    if value:
                        text_parts.append(value)
    result["chunks"] = chunks
    result["text"] = "".join(text_parts)
    return result


def report_result(result: dict[str, Any], done: int, total: int, args: argparse.Namespace) -> None:
    """Print one finished replay immediately.

    Called from the worker coroutine rather than after gather(), so progress is
    visible while a long replay is still running. flush=True because stdout is
    block-buffered when redirected to a file, which would otherwise defeat the
    point.

    With ``--show-response`` the generated text is printed underneath the status
    line, which is how a garbled or truncated generation becomes visible.
    """
    progress = f"[{done}/{total}]"
    if result.get("error"):
        # A failed request never got an id from the server, so the dumped one is
        # all there is to identify it by.
        line = f"[FAIL] {progress} {result['req_id']} attempt={result['attempt']} {result['error']}"
    else:
        extra = f" chunks={result['chunks']}" if result.get("streamed") else ""
        # Only the id the server just assigned: that is what to grep for in the
        # engine logs for this replay. The dumped id identifies the original
        # request and is not present on the node any more.
        line = (
            f"[ OK ] {progress} {result.get('new_req_id')} attempt={result['attempt']} "
            f"status={result['status']} completion_tokens={result.get('completion_tokens')}{extra}"
        )
    print(line, flush=True)

    if args.show_response and not result.get("error"):
        text = result.get("text") or ""
        if args.response_chars > 0:
            shown, clipped = text[: args.response_chars], len(text) > args.response_chars
        else:
            shown, clipped = text, False
        # Indent so the body stays visually attached to its status line even when
        # several requests are in flight.
        body = shown if shown else "(empty response)"
        for text_line in body.splitlines() or [""]:
            _print_safe(f"       | {text_line}")
        if clipped:
            print(f"       | ... [{len(text) - args.response_chars} more chars]", flush=True)


def _print_safe(line: str) -> None:
    """Print a line of model output without letting encoding kill the replay.

    Generated text is arbitrary, and a character the terminal's encoding cannot
    represent would raise UnicodeEncodeError. That exception would propagate out
    of the worker coroutine and abort every remaining request, so unrepresentable
    characters are escaped instead. This matters most when the generation is
    garbled, which is exactly what this flag is for.
    """
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(line.encode(encoding, errors="backslashreplace").decode(encoding), flush=True)


async def run(args: argparse.Namespace) -> int:
    records = load_records(args.input)
    if not records:
        print("nothing to replay", file=sys.stderr)
        return 1
    total_loaded = len(records)
    if args.limit > 0:
        records = records[: args.limit]
    print(
        f"replaying {len(records)} of {total_loaded} loaded request(s) "
        f"x {args.repeat} against {args.base_url}"
    )

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient(base_url=args.base_url, trust_env=args.trust_env) as client:
        if args.reset_prefix_cache:
            try:
                r = await client.post("/reset_prefix_cache", timeout=30.0)
                print(f"reset_prefix_cache -> {r.status_code}")
            except Exception as e:  # noqa: BLE001
                print(f"warning: reset_prefix_cache failed: {e}", file=sys.stderr)

        total = len(records) * args.repeat
        done = 0

        async def replay_and_report(rec: dict[str, Any], attempt: int) -> dict[str, Any]:
            nonlocal done
            result = await replay_one(client, rec, attempt, args, sem)
            # Incremented and printed here so each request is reported the moment
            # it finishes. Coroutines on one event loop do not interleave between
            # these two statements, so the counter needs no lock.
            done += 1
            report_result(result, done, total, args)
            return result

        tasks = [
            replay_and_report(rec, attempt)
            for attempt in range(1, args.repeat + 1)
            for rec in records
        ]
        results = await asyncio.gather(*tasks)

    failures = sum(1 for res in results if res.get("error"))
    print(f"\n{len(results) - failures}/{len(results)} replayed successfully")
    print("Check the decode node log for the SpecDecoding metrics line covering this window.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default="requests_*.jsonl",
        help="proxy request dump; glob patterns allowed (default: requests_*.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="replay only the first N loaded requests (0 = all)",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="proxy base url, e.g. http://127.0.0.1:9000 (not a decode node)",
    )
    parser.add_argument("--repeat", type=int, default=1, help="replay each request this many times")
    parser.add_argument("--concurrency", type=int, default=1, help="max in-flight requests")
    parser.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="print the generated text under each status line (off by default)",
    )
    parser.add_argument(
        "--response-chars",
        type=int,
        default=500,
        help="truncate each shown response to this many characters (0 = no limit)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="set max_tokens on every replayed request (0 = omit it and let the server decide). "
        "The dumped value is dropped either way because the proxy pins it to 1 for the prefill leg",
    )
    parser.add_argument(
        "--cache-salt",
        action="store_true",
        help="inject a unique cache_salt so the replay cannot hit the prefix cache "
        "(deviates from the original body; off by default)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="force stream=false instead of replaying the client's original stream setting",
    )
    parser.add_argument(
        "--reset-prefix-cache",
        action="store_true",
        help="POST /reset_prefix_cache on the proxy before replaying",
    )
    parser.add_argument(
        "--trust-env",
        action="store_true",
        help="honour http_proxy/https_proxy env vars (off by default so in-cluster traffic is not intercepted)",
    )
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
