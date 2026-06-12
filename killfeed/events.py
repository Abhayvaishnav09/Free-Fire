"""Killfeed event logging and JSONL output."""

from __future__ import annotations

import json
import time
from typing import Optional

from killfeed.queue_state import KillfeedEvent


class EventWriter:
    def __init__(
        self,
        log_path: str = "killfeed.log",
        jsonl_path: str = "killfeed_events.jsonl",
    ):
        self.log_path = log_path
        self.jsonl_path = jsonl_path

    def log_event(
        self,
        event: KillfeedEvent,
        frame: int = 0,
        sequence: int = 0,
    ) -> None:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(event.time))
        killer_display = event.killer or "ENV"
        line = (
            f"{ts} | {event.event} | {killer_display} → {event.victim} "
            f"| type={event.event_type} subtype={event.kill_type} "
            f"| icon={event.weapon} | hash={event.event_hash} | seq={sequence} | frame={frame}\n"
        )
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

        record = {
            "killer": event.killer,
            "victim": event.victim,
            "event": event.event,
            "event_type": event.event_type,
            "kill_type": event.kill_type,
            "weapon": event.weapon,
            "event_hash": event.event_hash,
            "position": event.position,
            "frame": frame,
            "sequence": sequence,
            "time": event.time,
        }
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

        print(
            f"✅ KILLFEED #{sequence}: {killer_display} → {event.victim} "
            f"[{event.event}] icon={event.weapon}"
        )
