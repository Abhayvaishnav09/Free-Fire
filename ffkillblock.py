"""Free Fire Killblock Detector - Main entry point for killfeed detection and API integration."""
import cv2
import os
import numpy as np
import warnings
import time
import tempfile
import datetime
from text import FreeFireTextDetector
import queue
import threading
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
    STATE_FILE = "match_state.json"
    GRPC_SERVER_PORT = 50051
    GRPC_SERVER_HOST = "localhost"
    
    def __init__(self, match_id="1", access_token=None, api_enabled=True, grpc_port=50051):
        print("Initializing Killblock Detector...")
        
        # gRPC configuration
        self.grpc_port = grpc_port
        self.grpc_channel = None
        self.grpc_stub = None
        self.server_process = None
        
        # Start gRPC server if not running
        if not self._check_server_running():
            print("🚀 Starting gRPC server...")
            if not self._start_grpc_server():
                print("❌ Failed to start gRPC server. Exiting...")
                return
        else:
            print(f"✅ gRPC server already running on port {self.grpc_port}")
        
        # Connect to gRPC server with retries
        max_connect_retries = 5
        for retry in range(max_connect_retries):
            if self._connect_to_server():
                break
            if retry < max_connect_retries - 1:
                print(f"⏳ Retrying connection ({retry+1}/{max_connect_retries})...")
                time.sleep(1)
            else:
                print("❌ Failed to connect to gRPC server after multiple attempts. Exiting...")
                return
        
        # Initialize OCR text detector
        try:
            self.text_detector = FreeFireTextDetector()
            print("✅ OCR text detector initialized")
        except (OSError, ImportError) as e:
            print(f"⚠️ Could not initialize text detector: {e}")
            print("⚠️ OCR functionality will be disabled. The script will continue but cannot extract names.")
            print("💡 To fix: Install Visual C++ Redistributables or reinstall PyTorch/PaddleOCR")
            self.text_detector = None
        except Exception as e:
            print(f"⚠️ Could not initialize text detector: {e}")
            self.text_detector = None
        
        self.recent_detections = {}
        self.duplicate_window = 8
        self.cap = None
        self.camera_initialized = False
        self.state_lock = threading.Lock()
        
        # API configuration
        self.match_id = match_id
        self.access_token = access_token
        self.api_enabled = api_enabled
        self.api_url = f'http://3.7.109.218:5005/LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed?matchId={match_id}'
        self.api_failed_count = 0
        
        state = self.load_state()
        self.detection_sequence = state.get('detection_sequence', 0)
        self.detected_count = state.get('detected_count', 0)
        self.saved_count = state.get('saved_count', 0)
        self.failed_count = state.get('failed_count', 0)
        self.frame_count_offset = state.get('frame_count_offset', 0)
        
        if self.detection_sequence > 0:
            print(f"📋 Resumed match state - Sequence: {self.detection_sequence}, Detected: {self.detected_count}, Saved: {self.saved_count}")
        
        self.save_queue = queue.Queue(maxsize=1000)
        self.save_thread = None
        self.save_thread_running = False
        
        if self.api_enabled:
            print(f"🌐 API integration enabled - Match ID: {match_id}")
        else:
            print("🌐 API integration disabled")
        
        print("Detector ready!")
    
    def __del__(self):
        self._cleanup_grpc()
        self.stop_save_thread()
        self.save_state()
        self.cleanup_camera()
    
    def _check_server_running(self):
        """Check if gRPC server is already running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((self.GRPC_SERVER_HOST, self.grpc_port))
            sock.close()
            return result == 0
        except Exception as e:
            return False
    
    def _start_grpc_server(self):
        """Start gRPC server in background. Server runs continuously until user stops it."""
        try:
            # Get the path to grpc_block.py
            grpc_block_path = os.path.join(os.path.dirname(__file__), 'grpc_block.py')
            if not os.path.exists(grpc_block_path):
                print(f"❌ grpc_block.py not found at {grpc_block_path}")
                return False
            
            print(f"🚀 Starting gRPC server process...")
            # Start server in a separate process with unbuffered output for better debugging
            # Server will run continuously until explicitly stopped
            self.server_process = subprocess.Popen(
                [sys.executable, '-u', grpc_block_path, '--serve', '--model', 'best.pt', '--port', str(self.grpc_port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                universal_newlines=True,
                bufsize=1,  # Line buffered
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
            )
            
            # Wait for server to start and check for errors
            max_retries = 20  # Increased retries for slower startup
            for i in range(max_retries):
                time.sleep(0.5)
                
                # Check if process died
                if self.server_process.poll() is not None:
                    # Server process died - read output to see why
                    try:
                        output, _ = self.server_process.communicate(timeout=1)
                        if output:
                            print(f"❌ Server process exited with output:")
                            print(output[:500])  # First 500 chars
                    except:
                        pass
                    return False
                
                # Check if server is listening
                if self._check_server_running():
                    print(f"✅ gRPC server started successfully on port {self.grpc_port}")
                    print(f"💡 Server will run continuously until you stop it (Ctrl+C)")
                    return True
                
                # Print progress every 5 attempts
                if (i + 1) % 5 == 0:
                    print(f"⏳ Waiting for server to start... ({i+1}/{max_retries})")
            
            # Check one more time if server is running
            if self._check_server_running():
                print(f"✅ gRPC server started successfully on port {self.grpc_port}")
                return True
            
            print(f"⚠️ Server may not be ready after {max_retries} attempts")
            print(f"💡 Checking server process status...")
            if self.server_process.poll() is None:
                print(f"✅ Server process is still running, attempting connection...")
                return True
            else:
                print(f"❌ Server process died")
                return False
        except Exception as e:
            print(f"❌ Failed to start gRPC server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _connect_to_server(self):
        """Connect to gRPC server."""
        try:
            if killfeed_detection_pb2 is None or killfeed_detection_pb2_grpc is None:
                print("❌ gRPC code not generated. Please run: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. killfeed_detection.proto")
                return False
            
            # Create channel with increased message size limits for large images
            options = [
                ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
                ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
            ]
            self.grpc_channel = grpc.insecure_channel(f'{self.GRPC_SERVER_HOST}:{self.grpc_port}', options=options)
            
            # Wait for channel to be ready
            try:
                grpc.channel_ready_future(self.grpc_channel).result(timeout=5)
            except grpc.FutureTimeoutError:
                print(f"⚠️ Timeout connecting to gRPC server on port {self.grpc_port}")
                return False
            
            # Create stub
            self.grpc_stub = killfeed_detection_pb2_grpc.KillfeedDetectionServiceStub(self.grpc_channel)
            print(f"✅ Connected to gRPC server on port {self.grpc_port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to gRPC server: {e}")
            return False
    
    def _cleanup_grpc(self):
        """Cleanup gRPC connection and server. Only called when user stops the client."""
        try:
            # Close channel (server continues running)
            if self.grpc_channel:
                self.grpc_channel.close()
                self.grpc_channel = None
                print("🔌 Disconnected from gRPC server")
            
            # Only stop server if we started it
            # Server will continue running if started manually
            if self.server_process:
                print("🛑 Stopping gRPC server (started by this client)...")
                try:
                    # Try graceful shutdown first
                    self.server_process.terminate()
                    self.server_process.wait(timeout=5)
                    print("✅ gRPC server stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    print("⚠️ Server didn't stop gracefully, forcing shutdown...")
                    self.server_process.kill()
                    self.server_process.wait(timeout=2)
                    print("✅ gRPC server force stopped")
                except Exception as e:
                    print(f"⚠️ Error stopping server: {e}")
                finally:
                    self.server_process = None
        except Exception as e:
            print(f"⚠️ Error cleaning up gRPC: {e}")
    
    def _process_frame_via_grpc(self, frame):
        """
        Process frame via gRPC server.
        
        Args:
            frame: Full captured frame (numpy array, BGR format)
        
        Returns:
            list: List of detection results (same format as before)
        """
        if self.grpc_stub is None or frame is None or frame.size == 0:
            return []
        
        try:
            # Encode frame to bytes
            _, frame_encoded = cv2.imencode('.png', frame)
            frame_bytes = frame_encoded.tobytes()
            
            # Create request
            request = killfeed_detection_pb2.ProcessFrameRequest(
                frame_image=frame_bytes,
                frame_width=frame.shape[1],
                frame_height=frame.shape[0]
            )
            
            # Call gRPC service with increased timeout for heavy processing
            response = self.grpc_stub.ProcessFrame(request, timeout=30)
            
            if not response.success:
                print(f"⚠️ gRPC server error: {response.error_message}")
                return []
            
            # Convert response to list format
            results = []
            for detection in response.detections:
                # Decode cropped image
                cropped_array = np.frombuffer(detection.cropped_image, dtype=np.uint8)
                cropped_image = cv2.imdecode(cropped_array, cv2.IMREAD_COLOR)
                
                if cropped_image is None or cropped_image.size == 0:
                    continue
                
                results.append({
                    'cropped_image': cropped_image,
                    'status': detection.status,
                    'bbox': [detection.bbox.x1, detection.bbox.y1, detection.bbox.x2, detection.bbox.y2],
                    'confidence': detection.confidence,
                    'detection_sources': list(detection.detection_sources)
                })
            
            return results
        except grpc.RpcError as e:
            error_code = e.code()
            error_details = e.details()
            
            if error_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                print(f"⚠️ gRPC timeout: Server took too long to process frame (>30s)")
                print(f"   Server is still running and processing - this frame timed out")
                print(f"   Skipping this frame and continuing... (server continues running)")
                return []
            elif error_code == grpc.StatusCode.UNAVAILABLE:
                print(f"⚠️ gRPC server unavailable: {error_details}")
                print("🔄 Attempting to reconnect to gRPC server...")
                if self._connect_to_server():
                    # Retry once
                    return self._process_frame_via_grpc(frame)
            else:
                print(f"⚠️ gRPC error: {error_code} - {error_details}")
            
            return []
        except Exception as e:
            print(f"⚠️ Error processing frame via gRPC: {e}")
            return []
    
    def load_state(self):
        """Load persistent match state from file. Returns empty dict if no state exists."""
        try:
            if os.path.exists(self.STATE_FILE):
                with open(self.STATE_FILE, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    if isinstance(state, dict):
                        return state
                    else:
                        print(f"⚠️ Invalid state file format, starting fresh")
                        return {}
            return {}
        except json.JSONDecodeError as e:
            print(f"⚠️ Corrupted state file, starting fresh: {e}")
            try:
                backup_name = f"{self.STATE_FILE}.corrupted_{int(time.time())}"
                os.rename(self.STATE_FILE, backup_name)
                print(f"📦 Backed up corrupted state to: {backup_name}")
            except:
                pass
            return {}
        except Exception as e:
            print(f"⚠️ Error loading state: {e}, starting fresh")
            return {}
    
    def save_state(self):
        """Save current match state to file. Thread-safe and atomic."""
        try:
            with self.state_lock:
                state = {
                    'detection_sequence': self.detection_sequence,
                    'detected_count': self.detected_count,
                    'saved_count': self.saved_count,
                    'failed_count': self.failed_count,
                    'frame_count_offset': self.frame_count_offset,
                    'last_saved': datetime.datetime.now().isoformat(),
                    'timestamp': time.time()
                }
                
                temp_file = f"{self.STATE_FILE}.tmp"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2, ensure_ascii=False)
                
                if os.path.exists(self.STATE_FILE):
                    os.replace(temp_file, self.STATE_FILE)
                else:
                    os.rename(temp_file, self.STATE_FILE)
        except Exception as e:
            print(f"⚠️ Error saving state: {e}")
    
    def reset_state(self):
        """Reset match state to start a new match. Use with caution."""
        try:
            with self.state_lock:
                self.detection_sequence = 0
                self.detected_count = 0
                self.saved_count = 0
                self.failed_count = 0
                self.frame_count_offset = 0
                self.save_state()
                print("🔄 Match state reset - starting new match")
        except Exception as e:
            print(f"⚠️ Error resetting state: {e}")
    
    def image_to_base64(self, image_path):
        """Convert an image file to base64 encoding."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"⚠️ Image file not found for base64 conversion: {image_path}")
            return None
        except Exception as e:
            print(f"⚠️ Error converting image to base64: {e}")
            return None

    def send_killblock_to_api(self, killer_name, victim_name, status, image_path, sequence_number):
        """Send killblock data to API. Returns True if successful, False otherwise."""
        if not self.api_enabled:
            return False
        
        try:
            base64_image = self.image_to_base64(image_path)
            if not base64_image:
                return False
            
            api_payload = {
                "killerName": killer_name,
                "victimName": victim_name,
                "WeaponUsed": status,  # API expects "WeaponUsed" field name with status value
                "imagePath": image_path,
                "siftWeapon": "",  # Empty for Free Fire (no weapon detection)
                "image": base64_image
            }
            
            headers = {
                'accept': 'text/plain',
                'Content-Type': 'application/json'
            }
            if self.access_token:
                headers['Authorization'] = f'Bearer {self.access_token}'
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=api_payload,
                timeout=10
            )
            
            if response.status_code == 200 or response.status_code == 201:
                print(f"✅ API: Successfully sent killblock #{sequence_number} to TMS")
                self.api_failed_count = 0
                return True
            else:
                print(f"⚠️ API: Failed to send killblock #{sequence_number} - Status: {response.status_code}")
                if response.text:
                    # Show full error response for better debugging
                    try:
                        error_json = response.json()
                        print(f"   Error Details: {json.dumps(error_json, indent=2)}")
                    except:
                        print(f"   Response: {response.text[:500]}")
                self.api_failed_count += 1
                return False
                
        except requests.exceptions.Timeout:
            print(f"⚠️ API: Timeout sending killblock #{sequence_number} to TMS")
            self.api_failed_count += 1
            return False
        except requests.exceptions.ConnectionError:
            print(f"⚠️ API: Connection error sending killblock #{sequence_number} to TMS")
            self.api_failed_count += 1
            return False
        except Exception as e:
            print(f"⚠️ API: Error sending killblock #{sequence_number} to TMS: {e}")
            self.api_failed_count += 1
            return False
    

    def process_detection_result(self, result, frame_number, sequence_number):
        """
        Process a detection result from grpc_block.
        
        Args:
            result: Dictionary with 'cropped_image', 'status', 'bbox', 'confidence', 'detection_sources'
            frame_number: Current frame number
            sequence_number: Detection sequence number
        
        Returns:
            bool: True if successfully processed, False otherwise
        """
        try:
            cropped_image = result.get('cropped_image')
            status = result.get('status', 'UNKNOWN')
            confidence = result.get('confidence', 0.0)
            
            if cropped_image is None or cropped_image.size == 0:
                return False
            
            # Extract names using OCR
            killer_name, victim_name = self.extract_names(cropped_image)
            
            # Check for REVIVED in OCR text (additional check)
            revive_text_detected = self.detect_revive_text(cropped_image)
            revive_in_names = False
            
            if not revive_text_detected:
                killer_upper = killer_name.upper().strip()
                victim_upper = victim_name.upper().strip()
                if 'REVIVED' in killer_upper or 'REVIVED' in victim_upper:
                    revive_in_names = True
                    print(f"✅ REVIVED found in extracted names: '{killer_name}' / '{victim_name}'")
                elif 'REVIVE' in killer_upper or 'REVIVE' in victim_upper:
                    killer_words = killer_upper.split()
                    victim_words = victim_upper.split()
                    if any(word in ['REVIVE', 'REVIVED'] for word in killer_words + victim_words):
                        revive_in_names = True
                        print(f"✅ REVIVED found in extracted names: '{killer_name}' / '{victim_name}'")
            
            # If OCR detects REVIVED, override status
            if revive_text_detected or revive_in_names:
                status = "REVIVED"
                print(f"✅ STATUS: REVIVED (detected via OCR text/names) - overriding status from grpc_block")
            
            # Check for duplicates
            if self.is_duplicate(killer_name, victim_name):
                print(f"🚫 Duplicate detected - skipping")
                return False
            
            # Prepare output
            output_dir = "cropkillblock"
            os.makedirs(output_dir, exist_ok=True)
            filename = f"{sequence_number:03d}_{killer_name} {status} {victim_name}.png"
            filepath = os.path.join(output_dir, filename)
            
            self.detected_count += 1
            metadata = {
                'killer_name': killer_name,
                'victim_name': victim_name,
                'status': status,
                'confidence': confidence,
                'sequence_number': sequence_number,
                'frame_number': frame_number
            }
            
            try:
                self.save_queue.put((cropped_image.copy(), filepath, metadata), timeout=0.1)
                print(f"📤 Enqueued for save: {filename}")
                print(f"   👤 Killer: {killer_name} | 🎯 Victim: {victim_name}")
                print(f"   📊 Status: {status} | 📈 Confidence: {confidence:.2f}")
                print(f"   🔢 Order: #{sequence_number} (Frame: {frame_number})")
                print(f"   💾 Queue size: {self.save_queue.qsize()} | Detected: {self.detected_count}\n")
                
                self.save_state()
                return True
            except queue.Full:
                print(f"⚠️ Save queue full, retrying...")
                try:
                    self.save_queue.put((cropped_image.copy(), filepath, metadata), timeout=1.0)
                    print(f"📤 Enqueued for save (retry): {filename}\n")
                    return True
                except queue.Full:
                    print(f"❌ Failed to enqueue - queue still full: {filename}\n")
                    self.failed_count += 1
                    return False
            except Exception as e:
                print(f"⚠️ Enqueue error: {e}")
                self.failed_count += 1
                return False
        except Exception as e:
            print(f"⚠️ Processing error: {e}")
            return False
    
    def detect_revive_text(self, cropped_image):
        """Detect REVIVED status by looking for 'REVIVED' text using OCR."""
        if cropped_image is None or cropped_image.size == 0 or self.text_detector is None:
            return False
        
        try:
            text_regions = self.text_detector.detect_text_regions(cropped_image)
            
            for region in text_regions:
                text = region.get('text', '').upper().strip()
                confidence = region.get('confidence', 0.0)
                original_text = region.get('text', '')
                
                text_clean = text.replace(' ', '').replace('-', '').replace('_', '').replace('.', '').replace(',', '')
                
                if 'REVIVED' in text or text.startswith('REVIVE'):
                    print(f"✅ REVIVED text detected via OCR: '{original_text}' (confidence: {confidence:.3f})")
                    return True
                
                revive_patterns = ['REVIVD', 'REVIV', 'REVIVE', 'REVIVED', 'REV1VED', 'REV1VE', 'REVIV3D']
                if any(pattern in text_clean for pattern in revive_patterns):
                    if len(text) <= 30:
                        print(f"✅ REVIVED text detected via OCR (variant): '{original_text}' (confidence: {confidence:.3f})")
                        return True
            
            return False
        except Exception as e:
            return False
    
    def extract_names(self, cropped_image):
        """Extract killer and victim names using OCR."""
        if self.text_detector is None:
            return "", ""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
            os.close(temp_fd)
            cv2.imwrite(temp_path, cropped_image)
            results = self.text_detector.process_image(temp_path)
            os.remove(temp_path)
            
            if results and isinstance(results, dict):
                killer = results.get("killer", {}).get("text", "") if isinstance(results.get("killer"), dict) else ""
                victim = results.get("victim", {}).get("text", "") if isinstance(results.get("victim"), dict) else ""
                return killer, victim
            return "", ""
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return "", ""

    def is_duplicate(self, killer_name, victim_name):
        """Check for duplicate detections."""
        try:
            current_time = time.time()
            old_sigs = [sig for sig, ts in self.recent_detections.items() if current_time - ts > self.duplicate_window]
            for sig in old_sigs:
                del self.recent_detections[sig]
            
            sig = f"{killer_name.lower()}_{victim_name.lower()}"
            if sig in self.recent_detections:
                return True
            self.recent_detections[sig] = current_time
            return False
        except:
            return False


    def initialize_camera(self):
        """Initialize OBS camera."""
        try:
            if self.cap is not None:
                self.cap.release()
            
            try:
                self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            except:
                self.cap = cv2.VideoCapture(1)
            
            if not self.cap.isOpened():
                print("Could not open OBS camera!")
                self.camera_initialized = False
                return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 1)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            for _ in range(5):
                self.cap.read()
            
            self.camera_initialized = True
            print("✅ Camera initialized")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.camera_initialized = False
            return False
    
    def capture_frame(self):
        """Capture frame from OBS with error recovery."""
        try:
            if not self.camera_initialized or self.cap is None:
                if not self.initialize_camera():
                    return None
            
            if self.cap is None:
                return None
            
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return None
            
            return frame
        except Exception as e:
            print(f"⚠️ Frame capture error: {e}")
            try:
                self.cleanup_camera()
                time.sleep(0.5)
                self.initialize_camera()
            except:
                pass
            return None
    
    def cleanup_camera(self):
        """Clean up camera resources."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.camera_initialized = False
        except:
            pass
    
    def save_worker_thread(self):
        """Background thread worker that saves images from the queue with retry logic."""
        max_retries = 3
        retry_delay = 0.1  # 100ms between retries
        
        while self.save_thread_running or not self.save_queue.empty():
            try:
                # Get item from queue with timeout to check running flag
                try:
                    item = self.save_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                image, filepath, metadata = item
                
                saved = False
                last_error = None
                for attempt in range(max_retries):
                    try:
                        if cv2.imwrite(filepath, image):
                            saved = True
                            self.saved_count += 1
                            break
                        else:
                            last_error = "cv2.imwrite returned False"
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay * (attempt + 1))
                    except Exception as e:
                        last_error = str(e)
                        if attempt < max_retries - 1:
                            print(f"⚠️ Save attempt {attempt+1}/{max_retries} failed: {e}, retrying...")
                            time.sleep(retry_delay * (attempt + 1))
                        else:
                            print(f"❌ Save failed after {max_retries} attempts: {e}")
                
                if saved:
                    print(f"📸 Saved: {os.path.basename(filepath)}")
                    print(f"   👤 Killer: {metadata.get('killer_name', '')} | 🎯 Victim: {metadata.get('victim_name', '')}")
                    print(f"   📊 Status: {metadata.get('status', '')} | 📈 Confidence: {metadata.get('confidence', 0.0):.2f}")
                    print(f"   🔢 Order: #{metadata.get('sequence_number', 0)} (Frame: {metadata.get('frame_number', 0)})")
                    print(f"   💾 Save Stats: {self.saved_count} saved / {self.detected_count} detected / {self.failed_count} failed\n")
                    
                    # Send to API after successful save
                    if self.api_enabled:
                        try:
                            self.send_killblock_to_api(
                                metadata.get('killer_name', ''),
                                metadata.get('victim_name', ''),
                                metadata.get('status', ''),
                                filepath,
                                metadata.get('sequence_number', 0)
                            )
                        except Exception as e:
                            print(f"⚠️ API: Unexpected error in API call: {e}")
                    
                    if self.saved_count % 5 == 0:
                        self.save_state()
                else:
                    self.failed_count += 1
                    error_msg = f": {last_error}" if last_error else ""
                    print(f"❌ Failed to save after {max_retries} retries{error_msg}: {os.path.basename(filepath)}")
                    print(f"   💾 Save Stats: {self.saved_count} saved / {self.detected_count} detected / {self.failed_count} failed\n")
                    if self.failed_count % 5 == 0:
                        self.save_state()
                
                self.save_queue.task_done()
                
            except Exception as e:
                print(f"⚠️ Save worker thread error: {e}")
                self.failed_count += 1
                try:
                    self.save_queue.task_done()
                except:
                    pass
                time.sleep(0.1)
    
    def start_save_thread(self):
        """Start the background save thread."""
        if self.save_thread is None or not self.save_thread.is_alive():
            self.save_thread_running = True
            self.save_thread = threading.Thread(target=self.save_worker_thread, daemon=True)
            self.save_thread.start()
            print("✅ Background save thread started")
    
    def stop_save_thread(self):
        """Stop the save thread and flush remaining items in queue."""
        if self.save_thread_running:
            print("🛑 Stopping save thread and flushing queue...")
            self.save_thread_running = False
            
            timeout = 30
            start_time = time.time()
            
            while not self.save_queue.empty():
                if time.time() - start_time > timeout:
                    print(f"⚠️ Queue flush timeout after {timeout}s, {self.save_queue.qsize()} items remaining")
                    break
                time.sleep(0.1)
            
            if self.save_thread is not None and self.save_thread.is_alive():
                self.save_thread.join(timeout=5)
                if self.save_thread.is_alive():
                    print("⚠️ Save thread did not terminate cleanly")
                else:
                    print("✅ Save thread stopped")
            
            print(f"📊 Final Save Stats: {self.saved_count} saved / {self.detected_count} detected / {self.failed_count} failed")
            self.save_state()

    def start_detection(self):
        """Main detection loop - runs continuously until user stops with Ctrl+C."""
        print("🚀 Starting Free Fire Detection...")
        print("="*50)
        print("⚠️  PRESS Ctrl+C TO STOP")
        print("📝 Will keep retrying if camera issues occur")
        print("="*50 + "\n")
        
        while not self.initialize_camera():
            print("⚠️ Camera initialization failed, retrying in 2 seconds...")
            time.sleep(2)
        
        self.start_save_thread()
        
        detection_count = 0
        frame_count = self.frame_count_offset
        last_detection_time = 0
        consecutive_failures = 0
        last_save_thread_check = time.time()
        save_thread_check_interval = 10
        last_state_save = time.time()
        state_save_interval = 30
        
        while True:
            try:
                current_time = time.time()
                if current_time - last_save_thread_check > save_thread_check_interval:
                    if self.save_thread is not None and not self.save_thread.is_alive():
                        print("⚠️ Save thread died! Restarting...")
                        self.start_save_thread()
                    last_save_thread_check = current_time
                
                if frame_count > 0 and frame_count % 200 == 0:
                    print(f"🔄 Periodic camera cleanup at frame #{frame_count}")
                    try:
                        self.cleanup_camera()
                        time.sleep(0.5)
                        if not self.initialize_camera():
                            print("⚠️ Camera reinit failed, will keep trying...")
                    except Exception as e:
                        print(f"⚠️ Cleanup error: {e}")
                
                frame = None
                for attempt in range(3):
                    try:
                        frame = self.capture_frame()
                        if frame is not None:
                            break
                    except Exception as e:
                        print(f"⚠️ Frame capture attempt {attempt+1}/3 failed: {e}")
                    
                    if frame is None:
                        time.sleep(0.3)
                
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures <= 5:
                        print(f"⚠️ Frame capture failed ({consecutive_failures} consecutive), retrying...")
                    else:
                        print(f"⚠️ Multiple capture failures ({consecutive_failures}), reconnecting camera...")
                        reconnect_attempts = 0
                        max_reconnect_attempts = 100
                        while reconnect_attempts < max_reconnect_attempts:
                            try:
                                self.cleanup_camera()
                                time.sleep(1)
                                if self.initialize_camera():
                                    print("✅ Camera reconnected successfully")
                                    consecutive_failures = 0
                                    break
                                reconnect_attempts += 1
                                if reconnect_attempts % 10 == 0:
                                    print(f"⚠️ Still trying to reconnect... ({reconnect_attempts}/{max_reconnect_attempts})")
                            except Exception as e:
                                print(f"⚠️ Reconnect attempt {reconnect_attempts} failed: {e}, retrying in 2 seconds...")
                                time.sleep(2)
                                reconnect_attempts += 1
                        
                        if reconnect_attempts >= max_reconnect_attempts:
                            print("⚠️ Max reconnect attempts reached, continuing with retry logic...")
                            consecutive_failures = 0
                    
                    time.sleep(0.5)
                    continue
                
                consecutive_failures = 0
                
                if frame.size == 0:
                    continue
                
                frame_count += 1
                time.sleep(0.3)  # Reduced delay for faster detection
                
                try:
                    # Use gRPC server to detect killfeeds and determine status
                    results = self._process_frame_via_grpc(frame)
                    current_time = time.time()
                    
                    if results and (current_time - last_detection_time) > 0.2:  # Reduced delay to catch more killfeeds
                        detection_count += 1
                        last_detection_time = current_time
                        
                        print(f"\n🎯 Detection #{detection_count} at frame #{frame_count}")
                        
                        for result in results:
                            try:
                                if len(results) > 1:
                                    time.sleep(0.001)
                                self.detection_sequence += 1
                                self.process_detection_result(result, frame_count, self.detection_sequence)
                            except Exception as e:
                                print(f"⚠️ Processing detection error: {e}")
                                continue
                        
                        print(f"✅ Processing complete\n")
                except Exception as e:
                    print(f"⚠️ Detection error: {e}")
                    continue
                
                if frame_count % 50 == 0:
                    print(f"⏳ Still running... Frame #{frame_count}")
                
                current_time_check = time.time()
                if current_time_check - last_state_save > state_save_interval:
                    self.frame_count_offset = frame_count
                    self.save_state()
                    last_state_save = current_time_check
            
            except KeyboardInterrupt:
                print("\n\n⏹️ Detection stopped by user (Ctrl+C)")
                self.frame_count_offset = frame_count
                self.save_state()
                break
            except Exception as e:
                print(f"⚠️ Loop error: {e}")
                import traceback
                traceback.print_exc()
                print("🔄 Attempting reconnect and continuing...")
                try:
                    self.cleanup_camera()
                    time.sleep(1)
                    self.initialize_camera()
                except Exception as reconnect_error:
                    print(f"⚠️ Reconnect error: {reconnect_error}")
                if self.save_thread is not None and not self.save_thread.is_alive():
                    print("⚠️ Save thread died during error recovery, restarting...")
                    self.start_save_thread()
                time.sleep(1)
                continue
        
        try:
            self.frame_count_offset = frame_count
            self.save_state()
            self.stop_save_thread()
            self.cleanup_camera()
        except:
            pass

def main():
    print("=== Free Fire Killblock Detector ===")
    
    # API configuration (optional)
    print("\n📡 API Configuration (optional):")
    print("Press Enter to use defaults (Match ID: 1, API enabled)")
    api_input = input("Enter Match ID (UUID or number, or press Enter for default 1): ").strip()
    match_id = api_input if api_input else "1"  # Accept UUID strings or numbers as strings
    
    token_input = input("Enter Access Token (or press Enter to skip): ").strip()
    access_token = token_input if token_input else None
    
    enable_api = input("Enable API integration? (Y/n, default: Y): ").strip().lower()
    api_enabled = enable_api != 'n'
    
    detector = KillblockDetector(match_id=match_id, access_token=access_token, api_enabled=api_enabled)
    
    if detector.grpc_stub is None:
        print("Failed to connect to gRPC server. Exiting...")
        return
    
    print("\nPlease start OBS Virtual Camera")
    print("Then press Enter to begin detection...")
    input()
    
    try:
        detector.start_detection()
    except KeyboardInterrupt:
        print("\n⏹️ User requested stop (Ctrl+C)")
        print("🛑 Stopping detection and cleaning up...")
    finally:
        try:
            detector.save_state()
        except:
            pass
        # Cleanup: stops server if we started it, closes connections
        # Server will continue running if started manually
        detector._cleanup_grpc()
        detector.stop_save_thread()
        detector.cleanup_camera()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
