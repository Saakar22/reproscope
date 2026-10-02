"""Pipeline event bus: every event is persisted (so jobs can be replayed) and
fanned out to live SSE subscribers. `emit` is safe to call from worker threads."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Optional

from sqlmodel import func, select

from .db import PipelineEvent, session, utcnow
from .schemas import EventOut

TERMINAL_EVENT = "job_finished"


class EventBus:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_seq: dict[str, int] = {}
        self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = {}

    # -------------------------------------------------------------- publish
    def _allocate_seq(self, job_id: str) -> int:
        if job_id not in self._next_seq:
            with session() as s:
                current = s.exec(select(func.max(PipelineEvent.seq))
                                 .where(PipelineEvent.job_id == job_id)).one()
            self._next_seq[job_id] = (current or 0) + 1
        seq = self._next_seq[job_id]
        self._next_seq[job_id] = seq + 1
        return seq

    def emit(self, job_id: str, message: str, *, stage: Optional[str] = None,
             level: str = "info", data: Any = None) -> EventOut:
        with self._lock:
            seq = self._allocate_seq(job_id)
            row = PipelineEvent(job_id=job_id, seq=seq, ts=utcnow(), stage=stage,
                                level=level, message=message, data=json.dumps(data))
            with session() as s:
                s.add(row)
                s.commit()
            out = EventOut(job_id=job_id, seq=seq, ts=row.ts, stage=stage,
                           level=level, message=message, data=data)
            subscribers = list(self._subscribers.get(job_id, []))
        for loop, queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, out)
            except RuntimeError:  # subscriber's loop already closed
                pass
        return out

    # ------------------------------------------------------------ subscribe
    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append((loop, queue))
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            self._subscribers[job_id] = [(l, q) for (l, q) in subs if q is not queue]

    # --------------------------------------------------------------- history
    @staticmethod
    def history(job_id: str, after_seq: int = 0) -> list[EventOut]:
        with session() as s:
            rows = s.exec(select(PipelineEvent)
                          .where(PipelineEvent.job_id == job_id, PipelineEvent.seq > after_seq)
                          .order_by(PipelineEvent.seq)).all()
        return [EventOut(job_id=r.job_id, seq=r.seq, ts=r.ts, stage=r.stage, level=r.level,
                         message=r.message, data=json.loads(r.data)) for r in rows]

    def forget(self, job_id: str) -> None:
        with self._lock:
            self._next_seq.pop(job_id, None)


bus = EventBus()


def sse_format(event: EventOut) -> str:
    """One SSE frame. `id` lets browsers resume with Last-Event-ID."""
    return f"id: {event.seq}\nevent: pipeline\ndata: {event.model_dump_json()}\n\n"
