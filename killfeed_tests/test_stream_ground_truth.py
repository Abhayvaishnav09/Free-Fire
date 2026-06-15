"""Compare stream ground truth to killfeed_events.jsonl (latest session)."""

from __future__ import annotations

import pytest

from killfeed_tests.test_helpers import (
    ROOT,
    coverage_report,
    find_in_detected,
    load_ground_truth,
    load_jsonl,
    normalize_detected,
)

JSONL_PATH = ROOT / "killfeed_events.jsonl"


def _latest_session_events(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    sessions: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        seq = int(row.get("sequence", 0))
        if seq <= 1 and current:
            sessions.append(current)
            current = [row]
        else:
            current.append(row)
    if current:
        sessions.append(current)
    return sessions[-1] if sessions else []


class TestStreamGroundTruthCoverage:
    @pytest.fixture(scope="class")
    def ground_truth(self):
        return load_ground_truth()

    @pytest.fixture(scope="class")
    def latest_detected(self):
        return _latest_session_events(load_jsonl(JSONL_PATH))

    def test_latest_session_exists(self, latest_detected):
        assert len(latest_detected) > 0, "killfeed_events.jsonl has no latest session"

    def test_stream_coverage_at_least_half(self, ground_truth, latest_detected):
        found, missing = coverage_report(ground_truth, latest_detected)
        ratio = len(found) / len(ground_truth)
        assert ratio >= 0.5, (
            f"Coverage {ratio:.0%} ({len(found)}/{len(ground_truth)}). Missing: {missing}"
        )

    def test_critical_knock_kill_pairs(self, latest_detected):
        pairs = [
            {"killer": "4END.SURYA", "victim": "EMZ.MAC", "event_type": "knock"},
            {"killer": "4END.SURYA", "victim": "EMZ.MAC", "event_type": "elimination"},
            {"killer": "EMZ.RUPESH", "victim": "4END.AVIJIT", "event_type": "knock"},
            {"killer": "EMZ.RUPESH", "victim": "4END.AVIJIT", "event_type": "elimination"},
            {"killer": "4END.SURYA", "victim": "EMZ.RUPESH", "event_type": "knock"},
            {"killer": "4END.SURYA", "victim": "EMZ.RUPESH", "event_type": "elimination"},
        ]
        missing = [p for p in pairs if not find_in_detected(p, latest_detected)]
        assert not missing, f"Missing critical pairs: {missing}"

    @pytest.mark.xfail(reason="Revive misclassified as knock/elimination")
    def test_revive_events(self, latest_detected):
        revives = [
            {"killer": "ARZ.PRODIGY", "victim": "ARZ.KUNAL10", "event_type": "revive"},
            {"killer": "GGI.PATLU", "victim": "GGI.POWER", "event_type": "revive"},
            {"killer": "TT.KHONSHU", "victim": "TT.KOWSIK24", "event_type": "revive"},
            {"killer": "4END.SURYA", "victim": "4END.AVIJIT", "event_type": "revive"},
        ]
        missing = [r for r in revives if not find_in_detected(r, latest_detected)]
        assert not missing, f"Missing revives: {missing}"

    def test_print_coverage_report(self, ground_truth, latest_detected, capsys):
        found, missing = coverage_report(ground_truth, latest_detected)
        print("\n=== STREAM GROUND TRUTH COVERAGE ===")
        print(f"Found: {len(found)}/{len(ground_truth)}")
        for m in missing:
            print(f"  MISS: {m['killer']} {m['event_type']} {m['victim']}")
        print(f"Latest session events: {len(latest_detected)}")
