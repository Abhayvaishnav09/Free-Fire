"""FIFO diff tests — burst killfeeds (3–4 at once) must not be missed."""

from __future__ import annotations

import pytest

from killfeed.queue_state import KillfeedState, find_new_slots, make_signature
from killfeed_tests.test_helpers import load_ground_truth, visible_snapshot


def sig(k: str, v: str, canonical: str = "normal_knock") -> str:
    return make_signature(k, v, canonical, "gun")


class TestFindNewSlotsBurst:
    """Unit tests for find_new_slots alignment / burst paths."""

    def test_three_new_while_top_ocr_stable(self):
        """Top row fuzzy-stable; 2 new rows below must not be skipped."""
        prev = [sig("A", "V1", "normal_knock")]
        cur = [
            sig("A", "V1", "normal_knock"),
            sig("B", "V2", "normal_knock"),
            sig("C", "V3", "normal_knock"),
        ]
        new = find_new_slots(prev, cur)
        assert len(new) == 2
        assert sig("B", "V2", "normal_knock") in new
        assert sig("C", "V3", "normal_knock") in new

    def test_four_burst_with_tail_alignment(self):
        prev = [sig("A", "V1"), sig("B", "V2")]
        cur = [
            sig("D", "V4"),
            sig("C", "V3"),
            sig("A", "V1"),
            sig("B", "V2"),
        ]
        new = find_new_slots(prev, cur)
        assert new == [sig("D", "V4"), sig("C", "V3")]

    def test_ocr_lag_four_new_scroll_off_previous(self):
        """OCR slow: 4 new rows appear before next snapshot; old row scrolled off."""
        prev = [sig("4END.SURYA", "EMZ.MAC", "normal_knock")]
        cur = [
            sig("GODL.NANCY", "TT.YOGESH23", "normal_knock"),
            sig("GODL.NANCY", "ARZ.JOHAN", "normal_elimination"),
            sig("GGI.SWARUP", "ARZ.MADGOD", "normal_knock"),
            sig("4END.SURYA", "4END.AVIJIT", "revive"),
        ]
        new = find_new_slots(prev, cur)
        assert len(new) == 3
        assert sig("4END.SURYA", "EMZ.MAC", "normal_knock") not in new

    def test_knock_to_kill_upgrade_at_top(self):
        prev = [sig("4END.SURYA", "EMZ.MAC", "normal_knock")]
        cur = [sig("4END.SURYA", "EMZ.MAC", "normal_elimination")]
        new = find_new_slots(prev, cur)
        assert new == [sig("4END.SURYA", "EMZ.MAC", "normal_elimination")]


class TestFifoSimulation:
    """Simulate on-screen snapshots event-by-event (ideal OCR)."""

    def test_incremental_stream_emits_all(self):
        events = load_ground_truth()
        state = KillfeedState(max_visible_slots=4, pair_cooldown_seconds=0.0)
        state.pair_cooldown.ttl_seconds = 0.0
        state.victim_cooldown.ttl_seconds = 0.0
        state.detected_cache.ttl_seconds = 0.0

        emitted_keys = []
        for i in range(len(events)):
            current = visible_snapshot(events, i, max_rows=4)
            for sig_new in state.diff_and_commit(current):
                k, v, canonical, _ = sig_new.split("|", 3)
                et = "revive" if canonical == "revive" else (
                    "elimination" if "elimination" in canonical else "knock"
                )
                emitted_keys.append((k.upper(), v.upper(), et))

        expected = [
            (e["killer"].upper(), e["victim"].upper(), e["event_type"])
            for e in events
        ]

        missing = [e for e in expected if e not in emitted_keys]
        assert not missing, f"FIFO simulation missed: {missing}"

    def test_burst_skip_three_frames_at_once(self):
        """YOLO/OCR lag: jump from 1 visible row to 4 new rows in one snapshot."""
        events = load_ground_truth()
        state = KillfeedState(max_visible_slots=4, pair_cooldown_seconds=0.0)
        state.pair_cooldown.ttl_seconds = 0.0
        state.victim_cooldown.ttl_seconds = 0.0
        state.detected_cache.ttl_seconds = 0.0

        state.diff_and_commit(visible_snapshot(events, 0))
        jumped = visible_snapshot(events, 4, max_rows=4)
        new = state.diff_and_commit(jumped)
        assert len(new) >= 3, f"Expected ≥3 new after lag burst, got {len(new)}: {new}"


class TestGroundTruthStreamRows:
    @pytest.mark.parametrize(
        "index,killer,victim,event_type",
        [
            (0, "4END.SURYA", "EMZ.MAC", "knock"),
            (1, "4END.SURYA", "EMZ.MAC", "elimination"),
            (11, "ARZ.PRODIGY", "ARZ.KUNAL10", "revive"),
            (19, "4END.SURYA", "4END.AVIJIT", "revive"),
            (22, "GODL.NANCY", "ARZ.JOHAN", "elimination"),
        ],
    )
    def test_fixture_rows(self, index, killer, victim, event_type):
        rows = load_ground_truth()
        row = rows[index]
        assert row["killer"] == killer
        assert row["victim"] == victim
        assert row["event_type"] == event_type
