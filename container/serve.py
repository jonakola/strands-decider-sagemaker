"""SageMaker serving shim for strands-decider, with continuous batching.

Wraps the package's FastAPI app (strands_decider.server.create_app) and adds the SageMaker
contract: GET /ping and POST /invocations on 0.0.0.0:8080. /invocations takes and returns
the System One shapes of /v1/systemone, plus `latency_ms` (enqueue to answer) and
`batch_size` (rows in the GPU pass that served it).

Single-question requests from different callers are batched into one forward (see
batching.py). Multi-question requests keep the engine's shared-prefix path.

Tuning (environment variables):
  MAX_BATCH          rows per forward                    (default 32)
  MAX_BATCH_TOKENS   rows x padded length per forward    (default 16384)
  MAX_DELAY_MS       how long a short batch may wait     (default 5)

Send `X-Amzn-SageMaker-Custom-Attributes: unbatched` to force the engine's own
one-request path (used to check batched answers against the reference).
"""

from __future__ import annotations

import json
import logging
import os
import time

import uvicorn
from fastapi import Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from batching import Batcher
from strands_decider import server
from strands_decider.schema import SystemOneRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("serve")

MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
CHECKPOINT = os.path.join(MODEL_DIR, "checkpoint")
MODEL_NAME = os.environ.get("MODEL_NAME", "strands-decider-2B-hobson-v19")

t0 = time.perf_counter()
app = server.create_app(CHECKPOINT, device="cuda", model_name=MODEL_NAME)
log.info("model loaded in %.1fs", time.perf_counter() - t0)

batcher = Batcher(
    server.get_engine(),
    max_batch=int(os.environ.get("MAX_BATCH", "32")),
    max_batch_tokens=int(os.environ.get("MAX_BATCH_TOKENS", "16384")),
    max_delay_ms=float(os.environ.get("MAX_DELAY_MS", "5")),
)


@app.get("/ping")
async def ping() -> dict:
    return {"status": "ok"}


@app.post("/invocations")
async def invocations(request: Request) -> JSONResponse:
    body = await request.body()
    try:
        req = SystemOneRequest.model_validate_json(body)
    except ValidationError as exc:
        return JSONResponse(status_code=400, content={"error": json.loads(exc.json())})
    attrs = request.headers.get("x-amzn-sagemaker-custom-attributes", "")
    try:
        job = await run_in_threadpool(batcher.prepare, req,
                                      force_unbatched="unbatched" in attrs)
        payload = await batcher.submit(job)
    except ValueError as exc:  # caller error, e.g. options truncated away
        return JSONResponse(status_code=422, content={"error": str(exc)})
    return JSONResponse(content=payload)


# Warm up before /ping answers: the first forwards compile the fla Triton kernels, and
# each new batch size can trigger another compile.
_WARM_STATE = "Help! My payouts have been failing for 3 days."
_WARM_Q = {"type": "choice", "instructions": "Route it.",
           "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}}
_multi = SystemOneRequest.model_validate({"state": _WARM_STATE, "questions": {
    "a": _WARM_Q, "b": {"type": "noul", "instructions": "Does this convey urgency?"}}})
t = time.perf_counter()
batcher.run_unbatched(_multi)
log.info("warmup multi-question %.0f ms", (time.perf_counter() - t) * 1000)
for n in (1, 2, 4, 8, 16, batcher.max_batch):
    rows = [batcher.prepare(SystemOneRequest.model_validate(
        {"state": f"{_WARM_STATE} Ticket {i}.", "questions": {"q": _WARM_Q}})).row
        for i in range(n)]
    t = time.perf_counter()
    out = batcher.run_rows(rows)  # type: ignore[arg-type]
    log.info("warmup batch=%d %.0f ms: %s", n, (time.perf_counter() - t) * 1000,
             json.dumps(out[0]["answers"]))
batcher.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1, log_level="info")
