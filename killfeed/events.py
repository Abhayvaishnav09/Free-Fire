"""Killfeed event logging and JSONL output."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from killfeed.queue_state import KillfeedEvent


class EventWriter:
    def __init__(
        self,
        log_path: str = "killfeed.log",
        jsonl_path: str = "killfeed_events.jsonl",
    ):
        self.log_path = os.path.abspath(log_path)
        self.jsonl_path = os.path.abspath(jsonl_path)

    def _append_line(self, path: str, line: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            print(f"⚠️ Failed to write killfeed log ({path}): {exc}")

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
        self._append_line(self.log_path, line)

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
        self._append_line(self.jsonl_path, json.dumps(record) + "\n")

        print(
            f"✅ KILLFEED #{sequence}: {killer_display} → {event.victim} "
            f"[{event.event}] icon={event.weapon}"
        )
