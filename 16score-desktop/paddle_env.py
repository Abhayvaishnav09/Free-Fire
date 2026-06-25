"""Paddle/PaddleOCR runtime env — import before paddleocr (used by ffkillblock.py)."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
