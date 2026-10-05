"""Dynamic batching for strands-decider single-question requests.

The upstream engine runs one request at a time: one state, N questions. A request with a
single question takes the engine's plain batched path (each row = state + question).
This module builds those same rows across *different* requests and runs them in one
forward, so concurrent single-question requests share a GPU pass.

Scheduling: one worker thread owns the GPU. It sleeps until a request arrives, then
takes every waiting single-question request (up to MAX_BATCH rows and MAX_BATCH_TOKENS
padded tokens) and runs them together. MAX_DELAY_MS lets a short batch wait a little for
company; under load the queue is never empty and the delay never applies. Multi-question
requests keep the engine's shared-prefix path and run alone, in arrival order.

Row construction reuses the engine's own `_fit` (truncation), `_pad`, `_temperatures`
and `_option_token_index`, so a batched row is the same tensor input the unbatched path
builds; only the batch neighbours differ. Pinned to strands-decider==0.1.0 because it
calls those private helpers.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from strands_decider.infer import SystemOneEngine, _option_token_index, _to_answer
from strands_decider.prompting import RenderedQuestion, render_question, render_state
from strands_decider.schema import SystemOneRequest, SystemOneResponse, Usage

log = logging.getLogger("batching")


@dataclass
class Row:
    """One single-question request, tokenised and ready to stack."""

    name: str
    rq: RenderedQuestion
    ids: list[int]
    opt: list[int] | None  # absolute option-token positions (pointer head), else None


@dataclass
class Job:
    request: SystemOneRequest
    row: Row | None  # None -> multi-question (or forced-unbatched) request
    loop: asyncio.AbstractEventLoop | None = None
    future: asyncio.Future | None = None
    enqueued: float = field(default_factory=time.perf_counter)


class Batcher:
    def __init__(self, engine: SystemOneEngine, *, max_batch: int = 32,
                 max_batch_tokens: int = 16384, max_delay_ms: float = 5.0) -> None:
        self.eng = engine
        self.max_batch = max_batch
        self.max_batch_tokens = max_batch_tokens
        self.max_delay = max_delay_ms / 1000
        self._queue: collections.deque[Job] = collections.deque()
        self._cond = threading.Condition()
        # `_fit` writes engine._last_offsets and the Rust tokenizer is not safe to share
        # across threads, so tokenisation is serialised (it is ~1 ms; the GPU is not).
        self._prep_lock = threading.Lock()
        self._pointer = engine.model.config.head_type == "pointer"
        self.stats = {"batches": 0, "rows": 0, "max_rows": 0}
        self._thread: threading.Thread | None = None

    # ---- request side ------------------------------------------------------

    def prepare(self, req: SystemOneRequest, *, force_unbatched: bool = False) -> Job:
        """Tokenise a request. Raises ValueError for caller errors, as the engine does."""
        if len(req.questions) != 1 or force_unbatched:
            return Job(req, None)
        name, q = next(iter(req.questions.items()))
        rq = render_question(q)
        cfg = self.eng.model.config
        if not self._pointer and rq.n_slots > cfg.num_slots:
            raise ValueError(
                f"question has {rq.n_slots} options but this model has {cfg.num_slots} slots"
            )
        state_text = render_state(req.state)
        with self._prep_lock:
            s, qs = self.eng._fit(state_text, [rq.text])
            offsets = self.eng._last_offsets[0]
        opt = None
        if self._pointer:
            # Same as engine._option_idx(rendered, base=len(s)), for one row.
            opt = [len(s) + i for i in _option_token_index(offsets, rq.option_spans, 0)]
        return Job(req, Row(name, rq, s + qs[0], opt))

    async def submit(self, job: Job) -> dict:
        job.loop = asyncio.get_running_loop()
        job.future = job.loop.create_future()
        with self._cond:
            self._queue.append(job)
            self._cond.notify()
        return await job.future

    # ---- GPU side ----------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="gpu-batcher", daemon=True)
        self._thread.start()

    def _take(self) -> list[Job]:
        with self._cond:
            while not self._queue:
                self._cond.wait()
            if self._queue[0].row is None:
                return [self._queue.popleft()]
            deadline = self._queue[0].enqueued + self.max_delay
            while len(self._queue) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            batch: list[Job] = []
            width = 0
            while self._queue and len(batch) < self.max_batch:
                nxt = self._queue[0]
                if nxt.row is None:
                    break  # leave multi-question requests for their own turn
                w = max(width, len(nxt.row.ids))
                if batch and w * (len(batch) + 1) > self.max_batch_tokens:
                    break
                width = w
                batch.append(self._queue.popleft())
            return batch

    def _loop(self) -> None:
        while True:
            jobs = self._take()
            try:
                if jobs[0].row is None:
                    payloads = [self.run_unbatched(jobs[0].request)]
                else:
                    payloads = self.run_rows([j.row for j in jobs])  # type: ignore[misc]
                    self.stats["batches"] += 1
                    self.stats["rows"] += len(jobs)
                    self.stats["max_rows"] = max(self.stats["max_rows"], len(jobs))
                    if self.stats["batches"] % 200 == 0:
                        log.info("batches=%(batches)d rows=%(rows)d max_rows=%(max_rows)d",
                                 self.stats)
            except Exception as exc:  # fail every request in the batch, keep serving
                log.exception("batch of %d failed", len(jobs))
                for j in jobs:
                    self._resolve(j, exc)
                continue
            for j, p in zip(jobs, payloads, strict=True):
                p["batch_size"] = len(jobs)
                p["latency_ms"] = round((time.perf_counter() - j.enqueued) * 1000, 2)
                self._resolve(j, p)

    @staticmethod
    def _resolve(job: Job, value: Any) -> None:
        def set_() -> None:
            if job.future is None or job.future.done():
                return
            if isinstance(value, Exception):
                job.future.set_exception(value)
            else:
                job.future.set_result(value)

        assert job.loop is not None
        job.loop.call_soon_threadsafe(set_)

    def run_unbatched(self, req: SystemOneRequest) -> dict:
        return self.eng.evaluate(req).model_dump()

    @torch.inference_mode()
    def run_rows(self, rows: list[Row]) -> list[dict]:
        """One forward over rows from different requests -> one response dict per row."""
        eng = self.eng
        ids, mask = eng._pad([r.ids for r in rows])
        opt_idx = None
        if self._pointer:
            w = max(len(r.opt) for r in rows)  # type: ignore[arg-type]
            opt_idx = torch.tensor(
                [r.opt + [-1] * (w - len(r.opt)) for r in rows],  # type: ignore[operator]
                dtype=torch.long, device=eng.device,
            )
        out = eng.model(
            input_ids=ids,
            attention_mask=mask,
            n_slots=torch.tensor([r.rq.n_slots for r in rows], device=eng.device),
            temperature=eng._temperatures([r.rq.kind for r in rows]),
            opt_idx=opt_idx,
        )
        probs = out["log_probs"].exp().float().cpu()
        ntok = mask.sum(dim=1).tolist()
        smoothing = eng.model.config.ordinal_smoothing
        results = []
        for i, r in enumerate(rows):
            ans = _to_answer(r.rq, probs[i, : r.rq.n_slots].tolist(),
                             ordinal_smoothing=smoothing)
            results.append(SystemOneResponse(
                model=eng.cfg.model_name,
                answers={r.name: ans},
                usage=Usage(input_tokens=int(ntok[i]), output_tokens=1),
            ).model_dump())
        return results
