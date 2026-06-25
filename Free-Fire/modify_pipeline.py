import re
import sys

def modify_pipeline():
    with open('/home/apandya/FreeFire-final-countdown/Free-Fire/killfeed/pipeline.py', 'r') as f:
        content = f.read()

    # 1. Add RowLedger imports
    content = content.replace(
        "from killfeed.queue_state import KillfeedState, find_new_slots, parse_signature, signatures_match",
        "from killfeed.queue_state import KillfeedState, find_new_slots, parse_signature, signatures_match\nfrom killfeed.row_ledger import RowLedger, TrackedRow"
    )

    # 2. Add RowLedger initialization in __init__
    init_replacement = """        self._overflow_lock = Lock()

        self.ledger = RowLedger(dhash_threshold=dhash_max_distance, ttl_frames=300)
        self._commit_thread = None
        self._tms_queue = Queue(maxsize=200)
        self._tms_thread = None"""
    
    content = content.replace("        self._overflow_lock = Lock()", init_replacement)

    # 3. Add to start()
    start_replacement = """        self._capture_thread.start()
        for t in self._ocr_threads:
            t.start()
        
        self._commit_thread = Thread(target=self._commit_loop, daemon=True, name="KillfeedCommit")
        self._commit_thread.start()
        self._tms_thread = Thread(target=self._tms_worker_loop, daemon=True, name="TMSPushWorker")
        self._tms_thread.start()"""
    
    content = content.replace("""        self._capture_thread.start()
        for t in self._ocr_threads:
            t.start()""", start_replacement)

    # 4. Replace _row_change_info with empty or remove it. We'll just remove it and use ledger in process_capture_frame.
    
    # 5. Rewrite _process_capture_frame
    process_capture_frame_new = """    def _process_capture_frame(self, bf, catchup_only: bool = False) -> None:
        with self._stats_lock:
            self._stats["frames_captured"] = bf.frame_num

        # Pre-YOLO Visual Gate
        if self._last_strip_image is not None and hasattr(self, '_roi_region_defined'):
            pass # can implement later

        roi = detect_strip(
            bf.frame,
            self.model,
            kill_confidence=self.kill_confidence,
            revive_confidence=self.revive_confidence,
            stabilizer=self._stabilizer,
            yolo_lock=self.yolo_lock,
        )
        
        sticky = False
        if roi is None or roi.strip.size == 0:
            with self._stats_lock:
                self._stats["yolo_misses"] += 1
            if (
                self._last_good_row_specs
                and self.yolo_sticky_frames > 0
                and (bf.frame_num - self._last_yolo_hit_frame) <= self.yolo_sticky_frames
            ):
                roi = reconstruct_strip_from_rows(
                    bf.frame, self._last_good_row_specs, self._stabilizer
                )
                if roi is not None and roi.strip.size > 0:
                    sticky = True
                    with self._stats_lock:
                        self._stats["sticky_recovery"] += 1
            if roi is None or roi.strip.size == 0:
                return
        else:
            with self._stats_lock:
                self._stats["yolo_hits"] += 1
            self._last_yolo_hit_frame = bf.frame_num
            self._store_row_specs(roi)

        row_count = len(roi.rows)
        
        # Use RowLedger instead of _row_change_info
        new_row_indices = []
        if not catchup_only:
            with self._snapshot_lock:
                new_row_indices = self.ledger.register_frame(roi.rows, bf.frame_num, bf.timestamp)
        
        mean_diff = 999.0
        if self._last_strip_image is not None:
            mean_diff = pixel_mean_diff(roi.strip, self._last_strip_image)

        if not catchup_only:
            self._frames_since_queue += 1
        heartbeat = (
            self.yolo_heartbeat_frames > 0
            and self._frames_since_queue >= self.yolo_heartbeat_frames
        )
        
        if heartbeat and not new_row_indices and self._pending_depth() > 3:
            with self._stats_lock:
                self._stats["gate_skips"] += 1
            return

        should_queue = (
            not catchup_only
            and (
                self.state.last_strip_dhash is None
                or sticky
                or bool(new_row_indices)
                or mean_diff >= self.roi_change_threshold
                or heartbeat
            )
        ) or (
            catchup_only
            and bool(new_row_indices)
        )

        if not should_queue:
            with self._stats_lock:
                self._stats["gate_skips"] += 1
            return

        if not catchup_only:
            self._last_row_count = row_count
            self._last_strip_image = roi.strip.copy()
            self._frames_since_queue = 0

        priority = 0
        if new_row_indices:
            priority = 10 if 0 in new_row_indices else 5
            
        self._enqueue_strip(
            StripJob(
                strip=roi.strip.copy(),
                roi=roi,
                frame_num=bf.frame_num,
                timestamp=bf.timestamp,
                new_row_indices=tuple(new_row_indices),
                priority=priority,
            )
        )"""

    # We need to replace the original _process_capture_frame up to _enqueue_strip
    # We will use regex
    pattern = re.compile(r'    def _process_capture_frame\(self, bf, catchup_only: bool = False\) -> None:.*?    def _enqueue_strip', re.DOTALL)
    content = pattern.sub(process_capture_frame_new + '\n\n    def _enqueue_strip', content)

    # 6. Replace _process_strip_job
    process_strip_job_new = """    def _process_strip_job(self, job: StripJob) -> None:
        try:
            with self._stats_lock:
                self._stats["ocr_runs"] += 1

            if not job.roi.rows:
                return

            job.roi.rows = consolidate_overlapping_rows(list(job.roi.rows))

            # Only process OCR for rows that were marked as new in the ledger
            # Or if it's a heartbeat, process all to update cache? 
            # Actually, the OCR workers should pick from ledger. But to minimize changes:
            # We will run OCR on the rows as before, but then update the ledger.
            
            with self._snapshot_lock:
                # We need to compute visual dhash for each row to update the ledger
                from killfeed.dhash import compute_dhash
                row_hashes = [compute_dhash(r.crop) for r in job.roi.rows]
                
                parsed_rows = self._parse_rows_parallel(
                    job.roi.rows, priority_indices=job.new_row_indices
                )
                if not parsed_rows:
                    self._maybe_gc()
                    return

                # Update ledger with OCR results
                for r, rh, pr in zip(job.roi.rows, row_hashes, parsed_rows):
                    if not pr: continue
                    if rh in self.ledger._rows:
                        tracked = self.ledger._rows[rh]
                        if not tracked.ocr_done:
                            tracked.ocr_done = True
                            tracked.ocr_killer = pr.killer
                            tracked.ocr_victim = pr.victim
                            tracked.ocr_confidence = pr.confidence
                            tracked.canonical = pr.canonical
                            tracked.tms_status = pr.tms_status
                            tracked.event_hash = pr.event.event_hash
                            # Cache the ParsedRow event for emission
                            tracked._parsed_row = pr
                            tracked._job = job
                
            self._maybe_gc()
        except Exception as exc:
            print(f"⚠️ Strip processing error: {type(exc).__name__}: {exc}")

    def _commit_loop(self):
        while not self._stop.is_set():
            with self._snapshot_lock:
                row = self.ledger.oldest_ready_to_emit()
                if row is None:
                    pass
                else:
                    pr = getattr(row, '_parsed_row', None)
                    job = getattr(row, '_job', None)
                    
                    if pr and job:
                        event_hash = row.event_hash
                        if self.state.should_emit(row.ocr_killer, row.ocr_victim, row.canonical, event_hash):
                            pr.event.frame = job.frame_num
                            pr.event.time = job.timestamp
                            # Fix position to be chronological based on arrival
                            pr.event.position = 1
                            pr.position = 1
                            
                            self.state.mark_emitted(row.ocr_killer, row.ocr_victim, row.canonical, event_hash)
                            seq = self.state.sequence
                            self.writer.log_event(pr.event, frame=job.frame_num, sequence=seq)
                            
                            try:
                                # push to async TMS queue instead of blocking
                                self._tms_queue.put_nowait((pr, job.frame_num, seq))
                            except Full:
                                print("⚠️ TMS queue full - dropping event")
                            
                            with self._stats_lock:
                                self._stats["events_emitted"] += 1
                        else:
                            with self._stats_lock:
                                self._stats["dup_skipped"] += 1
                                
                    self.ledger.mark_emitted(row.visual_dhash)
            
            time.sleep(0.01)

    def _tms_worker_loop(self):
        while not self._stop.is_set():
            try:
                pr, frame_num, seq = self._tms_queue.get(timeout=0.1)
                self.on_event(pr, frame_num, seq)
            except Empty:
                continue
            except Exception as e:
                print(f"⚠️ Async TMS Error: {e}")"""

    pattern2 = re.compile(r'    def _process_strip_job\(self, job: StripJob\) -> None:.*', re.DOTALL)
    content = pattern2.sub(process_strip_job_new, content)

    with open('/home/apandya/FreeFire-final-countdown/Free-Fire/killfeed/pipeline.py', 'w') as f:
        f.write(content)

if __name__ == "__main__":
    modify_pipeline()
    print("Done")
