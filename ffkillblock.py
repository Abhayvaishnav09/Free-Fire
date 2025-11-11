"""Free Fire Killblock Detector - Optimized client for killfeed detection and API integration."""
import cv2
import os
import numpy as np
import warnings
import time
import tempfile
import re
from collections import deque
from text import FreeFireTextDetector
import json
import base64
import requests
import subprocess
import sys
import socket
import grpc

# Import generated gRPC code
try:
    import killfeed_detection_pb2
    import killfeed_detection_pb2_grpc
except ImportError:
    print("⚠️ gRPC code not generated. Run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. killfeed_detection.proto")
    killfeed_detection_pb2 = None
    killfeed_detection_pb2_grpc = None

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('OMP_NUM_THREADS', '1')
warnings.filterwarnings("ignore")


class KillblockDetector:
    """Optimized killblock detector with three core responsibilities:
    1. Capture frames from OBS
    2. Send frames to gRPC server for processing
    3. Receive results, perform OCR, and send to TMS API
    """
    
    GRPC_SERVER_PORT = 50051
    GRPC_SERVER_HOST = "localhost"
    
    def __init__(self, match_id="1", access_token=None, api_enabled=True, grpc_port=50051):
        """Initialize detector with gRPC connection, OCR, and API configuration."""
        print("🚀 Initializing Killblock Detector...")
        
        # gRPC configuration
        self.grpc_port = grpc_port
        self.grpc_channel = None
        self.grpc_stub = None
        self.server_process = None
        
        # Start/connect to gRPC server
        if not self._check_server_running():
            print("🚀 Starting gRPC server...")
            if not self._start_grpc_server():
                raise RuntimeError("Failed to start gRPC server")
        else:
            print(f"✅ gRPC server already running on port {self.grpc_port}")
        
        # Connect to server
        if not self._connect_to_server():
            raise RuntimeError("Failed to connect to gRPC server")
        
        # Initialize OCR
        self.text_detector = self._init_ocr()
        
        # Camera
        self.cap = None
        
        # API configuration
        self.match_id = match_id
        self.access_token = access_token
        self.api_enabled = api_enabled
        self.api_url = f'http://3.7.109.218:5005/LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed?matchId={match_id}'
        
        # Detection tracking
        self.detection_sequence = 0
        self.recent_detections = set()  # Simple duplicate prevention
        
        # Temporal smoothing for OCR results
        self.name_history = deque(maxlen=5)  # Keep last 5 detections for voting
        self.confidence_threshold = 0.6  # Minimum confidence to accept
        self.min_name_length = 2  # Minimum name length
        self.max_name_length = 20  # Maximum name length
        
        # Known players dictionary (can be loaded from file or API)
        self.known_players = set()  # Will be populated from successful detections
        
        print("✅ Detector ready!")
    
    def _check_server_running(self):
        """Check if gRPC server is running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((self.GRPC_SERVER_HOST, self.grpc_port))
            sock.close()
            return result == 0
        except:
            return False
    
    def _start_grpc_server(self):
        """Start gRPC server in background."""
        try:
            grpc_block_path = os.path.join(os.path.dirname(__file__), 'grpc_block.py')
            if not os.path.exists(grpc_block_path):
                print(f"❌ grpc_block.py not found at {grpc_block_path}")
                return False
            
            self.server_process = subprocess.Popen(
                [sys.executable, '-u', grpc_block_path, '--serve', '--model', 'best.pt', '--port', str(self.grpc_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            # Wait for server to start
            for _ in range(20):
                time.sleep(0.5)
                if self._check_server_running():
                    print(f"✅ gRPC server started on port {self.grpc_port}")
                    return True
                if self.server_process.poll() is not None:
                    return False
            
            return self._check_server_running()
        except Exception as e:
            print(f"❌ Failed to start gRPC server: {e}")
            return False
    
    def _connect_to_server(self):
        """Connect to gRPC server."""
        try:
            if killfeed_detection_pb2 is None or killfeed_detection_pb2_grpc is None:
                print("❌ gRPC code not generated")
                return False
            
            options = [
                ('grpc.max_send_message_length', 50 * 1024 * 1024),
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),
            ]
            self.grpc_channel = grpc.insecure_channel(f'{self.GRPC_SERVER_HOST}:{self.grpc_port}', options=options)
            
            try:
                grpc.channel_ready_future(self.grpc_channel).result(timeout=5)
            except grpc.FutureTimeoutError:
                return False
            
            self.grpc_stub = killfeed_detection_pb2_grpc.KillfeedDetectionServiceStub(self.grpc_channel)
            print(f"✅ Connected to gRPC server on port {self.grpc_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def _init_ocr(self):
        """Initialize OCR text detector with simplified error handling."""
        print("🔍 Initializing OCR...")
        try:
            import paddleocr
            detector = FreeFireTextDetector()
            print("✅ OCR initialized successfully")
            return detector
        except ImportError:
            print("⚠️ PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
        except (OSError, Exception) as e:
            error_msg = str(e)
            if "shm.dll" in error_msg or "WinError 127" in error_msg:
                print("❌ Visual C++ Redistributables required")
                print("   Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
            else:
                print(f"⚠️ OCR initialization error: {e}")
        
        print("⚠️ OCR disabled - names will be empty")
        return None
    
    def _process_frame_via_grpc(self, frame):
        """Send frame to gRPC server and receive detection results."""
        if self.grpc_stub is None or frame is None or frame.size == 0:
            return []
        
        try:
            # Encode and send frame
            _, frame_encoded = cv2.imencode('.png', frame)
            request = killfeed_detection_pb2.ProcessFrameRequest(
                frame_image=frame_encoded.tobytes(),
                frame_width=frame.shape[1],
                frame_height=frame.shape[0]
            )
            
            # Get response
            response = self.grpc_stub.ProcessFrame(request, timeout=30)
            
            if not response.success:
                return []
            
            # Convert response to list
            results = []
            for detection in response.detections:
                cropped_array = np.frombuffer(detection.cropped_image, dtype=np.uint8)
                cropped_image = cv2.imdecode(cropped_array, cv2.IMREAD_COLOR)
                
                if cropped_image is None or cropped_image.size == 0:
                    continue
                
                results.append({
                    'cropped_image': cropped_image,
                    'status': detection.status,
                    'bbox': [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2],
                    'confidence': detection.confidence
                })
            
            return results
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print("⚠️ gRPC server unavailable, reconnecting...")
                self._connect_to_server()
            return []
        except Exception as e:
            print(f"⚠️ gRPC error: {e}")
            return []
    
    def _validate_image_for_ocr(self, image):
        """Validate and normalize image format for PaddleOCR."""
        if image is None or image.size == 0:
            return None
        
        try:
            # Ensure it's a numpy array
            if not isinstance(image, np.ndarray):
                return None
            
            # Handle different image formats
            if len(image.shape) == 2:
                # Grayscale - convert to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                if image.shape[2] == 4:
                    # RGBA - convert to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                elif image.shape[2] != 3:
                    return None
            else:
                return None
            
            # Check dimensions
            height, width = image.shape[:2]
            if height < 10 or width < 10:
                return None
            
            # PaddleOCR has size limits - resize if too large
            max_dimension = 2000  # PaddleOCR works best under 2000px
            if height > max_dimension or width > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Ensure uint8 type
            if image.dtype != np.uint8:
                if image.dtype == np.float32 or image.dtype == np.float64:
                    # Normalize to 0-255 range
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Ensure values are in valid range
            image = np.clip(image, 0, 255)
            
            return image
        except Exception as e:
            print(f"⚠️ Image validation error: {e}")
            return None
    
    def _preprocess_image_for_ocr(self, image, strategy='default', target_dpi=300):
        """
        Robust preprocessing with multiple strategies for maximum OCR accuracy.
        Returns validated image in correct format for PaddleOCR.
        """
        # First validate the input image
        image = self._validate_image_for_ocr(image)
        if image is None:
            return None
        
        try:
            height, width = image.shape[:2]
            processed = image.copy()
            
            # Strategy 1: Moderate upscaling (don't go too large - PaddleOCR has limits)
            if target_dpi > 0:
                # Limit upscaling to avoid memory issues
                scale_factor = min(2.5, max(1.5, target_dpi / 96.0))  # 1.5x to 2.5x max
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                # Ensure we don't exceed max dimensions
                if new_width <= 2000 and new_height <= 2000:
                    processed = cv2.resize(processed, (new_width, new_height), 
                                         interpolation=cv2.INTER_CUBIC)
            
            # Strategy 2: High contrast enhancement
            if strategy in ['default', 'high_contrast']:
                try:
                    lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                    l_enhanced = clahe.apply(l)
                    lab_enhanced = cv2.merge([l_enhanced, a, b])
                    processed = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
                except:
                    pass  # Fallback to original if CLAHE fails
            
            # Strategy 3: Black and white conversion
            if strategy == 'bw':
                try:
                    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                    # Adaptive thresholding for better text extraction
                    thresh = cv2.adaptiveThreshold(
                        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )
                    processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                except:
                    pass
            
            # Strategy 4: Sharpening
            if strategy in ['default', 'sharp']:
                try:
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                    processed = cv2.filter2D(processed, -1, kernel * 0.12)
                except:
                    pass
            
            # Strategy 5: Denoising
            if strategy == 'denoise':
                try:
                    processed = cv2.fastNlMeansDenoisingColored(processed, None, 8, 8, 7, 21)
                except:
                    pass
            
            # Final validation before returning
            processed = self._validate_image_for_ocr(processed)
            return processed
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            # Return validated original if preprocessing fails
            return self._validate_image_for_ocr(image)
    
    def _normalize_name(self, name):
        """Normalize name using regex rules and known players dictionary."""
        if not name or len(name) < self.min_name_length:
            return ""
        
        # Remove common OCR artifacts
        name = re.sub(r'[^\w\d._-]', '', name)  # Keep alphanumeric, dots, underscores, hyphens
        name = re.sub(r'_{2,}', '_', name)  # Replace multiple underscores with single
        name = name.strip('._-')  # Remove leading/trailing special chars
        
        # Fix common OCR mistakes
        replacements = {
            '0': 'O',  # In names, 0 is usually O
            '1': 'I',  # In names, 1 is usually I
            '5': 'S',  # Sometimes 5 is S
            '8': 'B',  # Sometimes 8 is B
        }
        # Only apply if it makes sense (not at start/end of numbers)
        for wrong, correct in replacements.items():
            # Replace standalone wrong chars (not part of numbers)
            name = re.sub(rf'(?<!\d){wrong}(?!\d)', correct, name)
        
        # Check against known players (fuzzy matching)
        if self.known_players:
            name_upper = name.upper()
            for known in self.known_players:
                known_upper = known.upper()
                # Exact match
                if name_upper == known_upper:
                    return known
                # Close match (1-2 char difference)
                if len(name_upper) >= 3 and len(known_upper) >= 3:
                    if self._levenshtein_distance(name_upper, known_upper) <= 2:
                        return known
        
        # Validate length
        if len(name) < self.min_name_length or len(name) > self.max_name_length:
            return ""
        
        return name
    
    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance for fuzzy matching."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _extract_names_with_ocr(self, cropped_image, preprocessing_strategy='default'):
        """Extract names using PaddleOCR with specified preprocessing and robust error handling."""
        if cropped_image is None or cropped_image.size == 0 or self.text_detector is None:
            return None, None, 0.0
        
        try:
            # Preprocess with specified strategy
            preprocessed = self._preprocess_image_for_ocr(cropped_image, strategy=preprocessing_strategy)
            
            if preprocessed is None:
                return None, None, 0.0
            
            # Validate preprocessed image
            if preprocessed.size == 0:
                return None, None, 0.0
            
            # Save to temp file with error handling
            temp_path = None
            try:
                temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                os.close(temp_fd)
                
                # Ensure image is valid before writing
                if not isinstance(preprocessed, np.ndarray):
                    return None, None, 0.0
                
                # Write with validation
                success = cv2.imwrite(temp_path, preprocessed, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                if not success or not os.path.exists(temp_path):
                    return None, None, 0.0
                
                # Run OCR with error handling
                results = self.text_detector.process_image(temp_path)
                
                if not results or not isinstance(results, dict):
                    return None, None, 0.0
                
                # Extract names and confidences
                killer_dict = results.get("killer", {})
                victim_dict = results.get("victim", {})
                
                killer = ""
                killer_conf = 0.0
                if isinstance(killer_dict, dict):
                    killer = killer_dict.get("text", "").strip()
                    killer_conf = float(killer_dict.get("confidence", 0.0))
                
                victim = ""
                victim_conf = 0.0
                if isinstance(victim_dict, dict):
                    victim = victim_dict.get("text", "").strip()
                    victim_conf = float(victim_dict.get("confidence", 0.0))
                
                # Calculate average confidence
                avg_conf = (killer_conf + victim_conf) / 2.0 if (killer or victim) else 0.0
                
                return killer, victim, avg_conf
                
            finally:
                # Clean up temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            
        except RuntimeError as e:
            # PaddleOCR specific runtime errors - try fallback
            error_msg = str(e).lower()
            if "unknown exception" in error_msg or "runtime" in error_msg:
                # Try with original image (no preprocessing) as fallback
                try:
                    original_validated = self._validate_image_for_ocr(cropped_image)
                    if original_validated is not None:
                        temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
                        os.close(temp_fd)
                        cv2.imwrite(temp_path, original_validated, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        results = self.text_detector.process_image(temp_path)
                        os.remove(temp_path)
                        
                        if results and isinstance(results, dict):
                            killer_dict = results.get("killer", {})
                            victim_dict = results.get("victim", {})
                            killer = killer_dict.get("text", "").strip() if isinstance(killer_dict, dict) else ""
                            victim = victim_dict.get("text", "").strip() if isinstance(victim_dict, dict) else ""
                            killer_conf = float(killer_dict.get("confidence", 0.0)) if isinstance(killer_dict, dict) else 0.0
                            victim_conf = float(victim_dict.get("confidence", 0.0)) if isinstance(victim_dict, dict) else 0.0
                            avg_conf = (killer_conf + victim_conf) / 2.0 if (killer or victim) else 0.0
                            return killer, victim, avg_conf
                except:
                    pass
            
            return None, None, 0.0
            
        except Exception as e:
            # Suppress verbose errors for failed strategies
            if preprocessing_strategy == 'default':
                print(f"⚠️ OCR error: {type(e).__name__}")
            return None, None, 0.0
    
    def _extract_names(self, cropped_image):
        """
        Robust name extraction with multiple preprocessing strategies, post-processing,
        and temporal smoothing. Only returns validated results.
        """
        if cropped_image is None or cropped_image.size == 0:
            return "", "", 0.0
        
        if self.text_detector is None:
            return "", "", 0.0
        
        # Try multiple preprocessing strategies
        strategies = ['default', 'high_contrast', 'bw', 'sharp', 'denoise']
        best_result = None
        best_confidence = 0.0
        
        for strategy in strategies:
            killer, victim, confidence = self._extract_names_with_ocr(cropped_image, strategy)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_result = (killer, victim, confidence)
            
            # If we got a high-confidence result, use it
            if confidence >= self.confidence_threshold:
                break
        
        if best_result is None:
            return "", "", 0.0
        
        killer, victim, confidence = best_result
        
        # Post-processing: normalize names
        killer = self._normalize_name(killer)
        victim = self._normalize_name(victim)
        
        # Validate both names are present and meet requirements
        if not killer or not victim:
            return "", "", 0.0
        
        if len(killer) < self.min_name_length or len(victim) < self.min_name_length:
            return "", "", 0.0
        
        # Temporal smoothing: vote with recent history
        smoothed_killer, smoothed_victim, smoothed_conf = self._apply_temporal_smoothing(
            killer, victim, confidence
        )
        
        return smoothed_killer, smoothed_victim, smoothed_conf
    
    def _apply_temporal_smoothing(self, killer, victim, confidence):
        """
        Apply temporal smoothing using voting across recent frames.
        High-confidence detections are carried forward.
        """
        # Add current detection to history
        detection = {
            'killer': killer,
            'victim': victim,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self.name_history.append(detection)
        
        # If confidence is high, use it directly
        if confidence >= self.confidence_threshold:
            # Update known players
            self.known_players.add(killer)
            self.known_players.add(victim)
            return killer, victim, confidence
        
        # If we have history, vote on most common names
        if len(self.name_history) >= 2:
            # Count occurrences of each name pair
            killer_votes = {}
            victim_votes = {}
            
            for hist in self.name_history:
                k = hist['killer']
                v = hist['victim']
                conf = hist['confidence']
                
                # Weight by confidence
                weight = conf
                
                killer_votes[k] = killer_votes.get(k, 0) + weight
                victim_votes[v] = victim_votes.get(v, 0) + weight
            
            # Get most voted names
            if killer_votes and victim_votes:
                best_killer = max(killer_votes.items(), key=lambda x: x[1])[0]
                best_victim = max(victim_votes.items(), key=lambda x: x[1])[0]
                
                # Use voted names if they're valid
                if (best_killer and best_victim and 
                    len(best_killer) >= self.min_name_length and 
                    len(best_victim) >= self.min_name_length):
                    
                    # Update known players
                    self.known_players.add(best_killer)
                    self.known_players.add(best_victim)
                    
                    # Return with boosted confidence
                    return best_killer, best_victim, min(1.0, confidence + 0.1)
        
        # Fallback: return current detection if valid
        if killer and victim:
            return killer, victim, confidence
        
        return "", "", 0.0
    
    def _save_image(self, image, filename):
        """Save cropped image to disk."""
        try:
            output_dir = "cropkillblock"
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, image)
            return filepath
        except Exception as e:
            print(f"⚠️ Save error: {e}")
            return None
    
    def _send_to_api(self, player_name, enemy_name, status, image_path, sequence_number):
        """Send killblock data to TMS API."""
        if not self.api_enabled:
            return False
        
        try:
            # Convert image to base64
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
            
            payload = {
                "killerName": player_name,  # player_name (victim in killfeed)
                "victimName": enemy_name,    # enemy_name (killer in killfeed)
                "WeaponUsed": status,
                "imagePath": image_path,
                "siftWeapon": "",
                "image": base64_image
            }
            
            headers = {'accept': 'text/plain', 'Content-Type': 'application/json'}
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=10)
            
            if response.status_code in [200, 201]:
                print(f"✅ API: Sent killblock #{sequence_number} to TMS")
                return True
            else:
                print(f"⚠️ API: Failed (Status: {response.status_code})")
                return False
        except Exception as e:
            print(f"⚠️ API error: {e}")
            return False
    
    def _validate_names(self, killer_name, victim_name, confidence):
        """Validate that names meet quality requirements before pushing to TMS."""
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False
        
        # Check both names are present
        if not killer_name or not victim_name:
            return False
        
        # Check name lengths
        if (len(killer_name) < self.min_name_length or 
            len(killer_name) > self.max_name_length or
            len(victim_name) < self.min_name_length or 
            len(victim_name) > self.max_name_length):
            return False
        
        # Check names are different
        if killer_name.upper() == victim_name.upper():
            return False
        
        # Check names don't contain only numbers
        if killer_name.isdigit() or victim_name.isdigit():
            return False
        
        return True
    
    def _process_detection(self, result):
        """
        Process a detection result: OCR, validate, save, and send to API.
        Only pushes to TMS when confidence and ordering are validated.
        """
        cropped_image = result.get('cropped_image')
        status = result.get('status', 'UNKNOWN')
        
        if cropped_image is None or cropped_image.size == 0:
            return False
        
        # Extract names using robust OCR (client-side)
        killer_name, victim_name, confidence = self._extract_names(cropped_image)
        
        # Validate names before proceeding
        if not self._validate_names(killer_name, victim_name, confidence):
            print(f"⚠️ Low confidence or invalid names (conf: {confidence:.2f}) - skipping")
            return False
        
        # Check for duplicates (simple set-based)
        detection_key = f"{killer_name}_{victim_name}_{status}"
        if detection_key in self.recent_detections:
            print(f"🚫 Duplicate detection - skipping")
            return False
        self.recent_detections.add(detection_key)
        if len(self.recent_detections) > 100:  # Limit set size
            self.recent_detections.clear()
        
        # Update sequence
        self.detection_sequence += 1
        
        # Save image (format: player_name → status → enemy_name)
        filename = f"{self.detection_sequence:03d}_{victim_name} {status} {killer_name}.png"
        filepath = self._save_image(cropped_image, filename)
        
        if not filepath:
            return False
        
        print(f"📸 Saved: {filename}")
        print(f"   👤 Player: {victim_name} | 🎯 Enemy: {killer_name} | 📊 Status: {status}")
        print(f"   ✅ Confidence: {confidence:.2f} | Validated: Yes")
        
        # Send to API (only validated results reach here)
        if self.api_enabled:
            success = self._send_to_api(victim_name, killer_name, status, filepath, self.detection_sequence)
            if success:
                # Add to known players for future normalization
                self.known_players.add(victim_name)
                self.known_players.add(killer_name)
        
        return True
    
    def _initialize_camera(self):
        """Initialize OBS camera."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            try:
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            except:
                self.cap = cv2.VideoCapture(1)
            
            if not self.cap.isOpened():
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 1)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Flush buffer
            for _ in range(5):
                self.cap.read()
            
            return True
        except:
            return False
    
    def _capture_frame(self):
        """Capture frame from OBS camera."""
        try:
            if self.cap is None or not self.cap.isOpened():
                if not self._initialize_camera():
                    return None
            
            ret, frame = self.cap.read()
            return frame if ret else None
        except:
            return None
    
    def start_detection(self):
        """Main detection loop: capture → gRPC → OCR → API."""
        print("\n🚀 Starting detection loop...")
        print("⚠️  Press Ctrl+C to stop\n")
        
        # Initialize camera
        while not self._initialize_camera():
            print("⚠️ Camera initialization failed, retrying in 2 seconds...")
            time.sleep(2)
        
        print("✅ Camera ready")
        frame_count = 0
        last_detection_time = 0
        
        try:
            while True:
                # Capture frame
                frame = self._capture_frame()
                if frame is None or frame.size == 0:
                    time.sleep(0.5)
                    continue
                
                frame_count += 1
                time.sleep(0.3)  # Frame rate control
                
                # Send to gRPC server
                results = self._process_frame_via_grpc(frame)
                
                # Process detections
                if results:
                    current_time = time.time()
                    if current_time - last_detection_time > 0.2:  # Prevent duplicates
                        last_detection_time = current_time
                        print(f"\n🎯 Detection at frame #{frame_count}")
                        
                        for result in results:
                            self._process_detection(result)
                        
                        print("✅ Processing complete\n")
                
                if frame_count % 50 == 0:
                    print(f"⏳ Running... Frame #{frame_count}")
        
        except KeyboardInterrupt:
            print("\n⏹️ Detection stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources."""
        try:
            if self.cap is not None:
                self.cap.release()
            if self.grpc_channel:
                self.grpc_channel.close()
            if self.server_process:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
        except:
            pass
        print("✅ Cleanup complete")


def main():
    """Main entry point."""
    print("=== Free Fire Killblock Detector ===")
    
    # Get configuration
    match_id = input("Enter Match ID (or press Enter for default '1'): ").strip() or "1"
    access_token = input("Enter Access Token (or press Enter to skip): ").strip() or None
    api_enabled = input("Enable API integration? (Y/n, default: Y): ").strip().lower() != 'n'
    
    try:
        detector = KillblockDetector(match_id=match_id, access_token=access_token, api_enabled=api_enabled)
        
        if detector.grpc_stub is None:
            print("❌ Failed to connect to gRPC server")
            return
        
        print("\n📹 Please start OBS Virtual Camera")
        input("Press Enter to begin detection...")
        
        detector.start_detection()
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
