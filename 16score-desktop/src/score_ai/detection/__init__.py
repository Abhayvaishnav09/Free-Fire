"""
Detection module for 16Score AI application.
Contains kill feed detection and processing functionality.

Modules:
    - killfeed_detections: Detection data storage and management
    - detection_pb2: Protocol buffer definitions for gRPC killfeed service
    - detection_pb2_grpc: gRPC service stubs for killfeed detection
    - killblocks: Main detection logic with SIFT matching
"""

from .killfeed_detections import KillfeedDetections

# gRPC protocol buffer imports
from . import detection_pb2, detection_pb2_grpc
from . import killfeed_pb2, killfeed_pb2_grpc

__all__ = [
    'KillfeedDetections',
    'detection_pb2',
    'detection_pb2_grpc',
    'killfeed_pb2',
    'killfeed_pb2_grpc',
] 