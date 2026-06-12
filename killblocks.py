import cv2
import numpy as np
import time
import os
import base64
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pymongo import MongoClient 
import json
from rapidfuzz import fuzz, process
from difflib import SequenceMatcher
import logging
import glob
import threading
import traceback
from datetime import datetime
import torch
from killfeed_detections import KillfeedDetections
from collections import deque, defaultdict
from app.utils.resource_utils import resource_path

# Check if GPU is available using PyTorch instead of OpenCV CUDA
def check_opencv_gpu():
    """Check if GPU is available for image processing acceleration using PyTorch."""
    try:
        # Check if PyTorch can access GPU
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available: {device_name}")
            print(f"CUDA device count: {device_count}")
            torch.cuda.set_device(0)  # Use the first GPU
            return True
        else:
            print("GPU acceleration not available (no compatible GPU found or drivers not properly installed)")
            return False
    except Exception as e:
        print(f"Error checking GPU support: {str(e)}")
        return False

# Global flag for GPU usage
USE_GPU = check_opencv_gpu()

# Global cache for SIFT features
sift_template_cache = {}

# Global queue for last 5 killfeeds
recent_killfeeds = deque(maxlen=5)

def cache_sift_templates():
    """
    Precompute and cache SIFT keypoints and descriptors for all templates in single-template.
    """
    sift = cv2.SIFT_create(nfeatures=400)
    weapon_folder = resource_path('app/single-template')
    for subdir, dirs, files in os.walk(weapon_folder):
        for file in files:
            template_image_path = os.path.join(subdir, file)
            template_image = cv2.imread(template_image_path)
            if template_image is None:
                continue
            gray_template = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            gray_template = cv2.resize(gray_template, (0, 0), fx=0.75, fy=0.75)
            keypoints, descriptors = sift.detectAndCompute(gray_template, None)
            if descriptors is not None:
                template_name = os.path.splitext(file)[0].split('-')[0]
                # Store as a list in case of multiple templates per weapon
                if template_name not in sift_template_cache:
                    sift_template_cache[template_name] = []
                sift_template_cache[template_name].append((descriptors, template_image_path))

# Call this at startup
cache_sift_templates()

def process_frame(frame, local_model, ocr, killfeed_detector, TEAM_ROSTERS, 
                 Match_name, collection, game_logger, match_id, access_token, USE_GPU, device, frame_counter, position):
    """
    Process a single frame to detect kill events.
    """
    confidence_threshold = 0.20
    api_call_count = 0

    kill_block_folder = f"{Match_name}/kill_block"
    os.makedirs(kill_block_folder, exist_ok=True)

    try:
        # Convert frame to grayscale for better processing
        if USE_GPU:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float().to(device) / 255.0
            gray_tensor = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            gray_frame = (gray_tensor.cpu().numpy() * 255).astype(np.uint8)
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # First check for kill blocks using local YOLO model
        local_results = local_model(frame, classes=[12])
        
        # Only proceed with API call if kill blocks are detected
        if len(local_results[0].boxes) > 0:
            api_call_count += 1

            temp_frame_path = "temp_frame.jpg"
            cv2.imwrite(temp_frame_path, gray_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            try:
                with open(temp_frame_path, 'rb') as img_file:
                    api_response = requests.post(
                        'http://192.168.29.224:8000/api/v1/yolo/predict',
                        files={'file': ('temp_frame.jpg', img_file, 'image/jpeg')}
                    )
                    
                if api_response.status_code != 200:
                    return position
                    
                response_data = api_response.json()
                
                if not response_data.get('success'):
                    return position
                    
                detections = response_data.get('detections', [])
                
                if not detections:
                    return position
                
                kill_blocks = []
                other_objects = []
                
                for detection in detections:
                    try:
                        confidence = detection.get('confidence', 0)
                        if confidence >= confidence_threshold:
                            class_id = detection.get('class_id')
                            class_name = detection.get('class_name')
                            bbox = detection.get('bbox')
                            
                            if not bbox or len(bbox) != 4: 
                                continue
                            x1, y1, x2, y2 = map(int, bbox)
                            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1: 
                                continue
                                
                            detection_data = {
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': bbox,
                                'cropped_image_url': detection.get('cropped_image_url')
                            }
                            killfeed_detector.add_detection(detection_data)
                            
                            if class_id == 12 or class_name == 'kill-block':
                                kill_blocks.append((x1, y1, x2, y2))
                                if detection.get('cropped_image_url'):
                                    try:
                                        cropped_img_response = requests.get(detection.get('cropped_image_url'))
                                        if cropped_img_response.status_code == 200:
                                            kb_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                            kb_filename = f"frame_{frame_counter}_{kb_timestamp}_kill_block.jpg"
                                            kb_path = os.path.join(kill_block_folder, kb_filename)
                                            with open(kb_path, 'wb') as f:
                                                f.write(cropped_img_response.content)
                                    except Exception as e:
                                        continue
                            else:
                                other_objects.append((x1, y1, x2, y2, class_name))
                    except Exception as e:
                        continue
                        
                kill_blocks.sort(key=lambda x: x[1])
                
                if not kill_blocks:
                    return position

                for kb_index, kb_coords in enumerate(kill_blocks):
                    try:
                        x1_kb, y1_kb, x2_kb, y2_kb = kb_coords
                        if y2_kb > gray_frame.shape[0] or x2_kb > gray_frame.shape[1]: 
                            continue
                        
                        kill_block_region = gray_frame[y1_kb:y2_kb, x1_kb:x2_kb]
                        if kill_block_region.size == 0: 
                            continue
                        
                        kb_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        detected_text_boxes = text_detection(kill_block_region, ocr)
                        
                        # Save kill block image
                        kill_block_path = os.path.join(kill_block_folder, f"frame_{frame_counter}_{kb_index}_{kb_timestamp}_kill_block.jpg")
                        cv2.imwrite(kill_block_path, kill_block_region, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        
                        if not detected_text_boxes:
                            detected_text_boxes = []
                        else:
                            detected_text_boxes.sort(key=lambda box: box[0])
                        
                        killer_name, victim_name, weapon_used = extract_kill_info(
                            detected_text_boxes, other_objects, x1_kb, y1_kb, x2_kb, y2_kb)
                        
                        # Check if weapon is a kill or knockout type
                        is_kill_or_knockout = any(keyword in weapon_used.lower() for keyword in ['kill', 'knockout'])
                        
                        if not is_kill_or_knockout:
                            try: os.remove(kill_block_path)
                            except: pass
                            continue
                        
                        sift_result, sift_weapon_name = verify_weapon_with_sift(kill_block_folder, weapon_used)
                        
                        killer_name = get_player_team(killer_name, TEAM_ROSTERS)
                        victim_name = get_player_team(victim_name, TEAM_ROSTERS)
                        
                        position = save_to_mongodb(
                            killer_name, weapon_used, victim_name, kill_block_path, 
                            (sift_result, sift_weapon_name), position, collection, match_id, access_token, 
                            frame_timestamp=kb_timestamp)
                            
                    except Exception as e:
                        continue
            except Exception as e:
                return position
    except Exception as e:
        return position
    
    return position

def obs_frame_capture(match_id=1, access_token=None, camera_index=1):
    """
    Main function to capture frames from OBS and detect kill events in game footage.
    Processes frames every 0.2 seconds.
    
    Args:
        match_id (int): ID of the match being processed
        access_token (str): Authentication token for API access
        camera_index (int): Index of the camera to use for capture
    """
    print(f"Processing match ID: {match_id}")
    print(f"Using camera index: {camera_index}")
    
    match_data = fetch_match_data(match_id, access_token)
    if not match_data:
        return
    
    Match_name = match_data.get("collectionName")
    db_name = match_data.get("databaseName")

    uri = "mongodb://192.168.29.47:27017/"
    client = MongoClient(uri)
    db = client[db_name]
    collection = db[Match_name]

    local_model = YOLO(resource_path('app/best_50.pt'))
    local_model.conf = 0.20

    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, det_db_box_thresh=0.4, drop_score=0.3)
    killfeed_detector = KillfeedDetections(Match_name)
    
    game_logger = setup_logger(Match_name)
    game_logger.info(f"Starting new session for match ID: {match_id}")
    game_logger.info(f"Using camera index: {camera_index}")

    TEAM_ROSTERS = load_team_rosters_from_api(match_id, access_token)
    TEAM_ROSTERS = TEAM_ROSTERS.get("teams", {})
    game_logger.info(f"Loaded team rosters: {TEAM_ROSTERS}")

    device = torch.device("cuda" if USE_GPU else "cpu")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    last_saved_time = time.time()
    frame_capture_interval = 0.2
    frame_counter = 0
    position = 1

    # Main capture loop
    while True:
        try:
            current_time = time.time()
            if current_time - last_saved_time >= frame_capture_interval:
                last_saved_time = current_time
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.1)
                    continue
                
                # Process the frame
                position = process_frame(frame, local_model, ocr, killfeed_detector, TEAM_ROSTERS,
                                       Match_name, collection, game_logger, match_id, access_token, 
                                       USE_GPU, device, frame_counter, position)
                frame_counter += 1
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' pressed, stopping capture...")
                break
        except KeyboardInterrupt:
            print("Interrupted by user, stopping capture...")
            break
        except Exception as e:
            time.sleep(0.1)
            continue

    # Clean up resources
    print("Shutting down...")
    cap.release()
    cv2.destroyAllWindows()
    print("Shutdown complete.")

def fetch_match_data(match_id, access_token=None):
    """
    Fetches match data from the API.
    
    Args:
        match_id (int): The ID of the match to fetch
        access_token (str): Authorization token for the API
        
    Returns:
        dict: Match data or None if error occurred
    """
    try:
        # Default token if none provided
        if not access_token:
            access_token = ""
            print("No access token provided")
            
        headers = {'Authorization': f'Bearer {access_token}'} if access_token else {}
        
        response = requests.get(
            f'http://192.168.29.47:81/LeagueMatchData/LeagueMatch/info?matchId={match_id}',
            headers=headers
        )
        response.raise_for_status()
        
        if not response.text:
            print("Error: Empty response from API")
            return None
            
        data = response.json()
        if not data or "data" not in data:
            print("Error: Invalid response format from API")
            return None
            
        return data.get("data")
            
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
    
    return None

def setup_logger(match_name):
    """
    Sets up a logger for recording game events.
    
    Args:
        match_name (str): Name of the match for log file naming
        
    Returns:
        Logger: Configured logger object
    """
    log_folder = "game_logs"
    os.makedirs(log_folder, exist_ok=True)
    log_filename = os.path.join(log_folder, f"team_stats_{match_name}.log")

    # Create and configure file handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Set up the logger
    game_logger = logging.getLogger('game_stats')
    game_logger.setLevel(logging.INFO)
    game_logger.propagate = False

    # Remove any existing handlers to avoid duplication
    game_logger.handlers.clear()
    game_logger.addHandler(file_handler)

    return game_logger

def load_team_rosters_from_api(match_id, access_token=None):
    """
    Load team roster data from API.
    
    Args:
        match_id (int): Match ID to get team data for
        access_token (str): Authentication token for API access
        
    Returns:
        dict: Team roster data in a structured format
    """
    try:
        # Set up headers with access token if provided
        headers = {'Authorization': f'Bearer {access_token}'} if access_token else {}
        
        # Make API request to get team data
        api_url = f"http://192.168.29.47:81/LeagueMatchData/LeagueMatch/LeagueMatchId/teams-players?matchId={match_id}"
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        
        # Parse response
        data = response.json()
        
        # Check if response is valid
        if not data or not data.get("success") or not data.get("data"):
            print(f"API Error: Invalid team data response: {data}")
            return {"teams": {}}
        
        # Format data into team roster structure
        formatted_teams_data = {"teams": {}}
        
        # Process each team
        for team in data.get("data", []):
            team_name = team.get("team_name")
            if not team_name:
                continue
                
            # Get player data
            players = {key: value for key, value in team.items() if key.startswith("player") and value}
            formatted_teams_data["teams"][team_name] = players
            
        print(f"Retrieved {len(formatted_teams_data['teams'])} teams from API")
        print(formatted_teams_data)
        return formatted_teams_data
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team roster from API: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"Error parsing team roster API response: {str(e)}")
    except Exception as e:
        print(f"Unexpected error in team roster API call: {str(e)}")
    
    return {"teams": {}}

def string_similarity(a, b):
    """
    Calculate similarity ratio between two strings using RapidFuzz.
    
    Args:
        a (str): First string to compare
        b (str): Second string to compare
        
    Returns:
        float: Similarity ratio between 0 and 1
    """
    # Handle invalid input types
    if isinstance(a, tuple) or isinstance(b, tuple):
        return 0.0

    if not a or not b:
        return 0.0
    
    # Normalize strings for comparison
    str1 = a.lower().strip()
    str2 = b.lower().strip()
    
    # Use RapidFuzz for better and faster string comparison
    # Calculate weighted ratio which handles partial string matches better
    return fuzz.ratio(str1, str2) / 100.0

def string_similarity_original(a, b):
    """
    Calculate similarity ratio between two strings using SequenceMatcher (original method).
    
    Args:
        a (str): First string to compare
        b (str): Second string to compare
        
    Returns:
        float: Similarity ratio between 0 and 1
    """
    # Handle invalid input types
    if isinstance(a, tuple) or isinstance(b, tuple):
        return 0.0

    if not a or not b:
        return 0.0
    
    # Normalize strings for comparison
    str1 = a.lower().strip()
    str2 = b.lower().strip()
    
    # Use SequenceMatcher for string comparison
    matcher = SequenceMatcher(None, str1, str2)
    return matcher.ratio()

def get_player_team(player_name, team_rosters):
    """
    Find the team a player belongs to based on name similarity.
    
    Args:
        player_name (str): Player name to look up
        team_rosters (dict): Dictionary of team rosters
        
    Returns:
        str: The matching player name from team rosters, or "unknown" if no good match found
    """
    # Handle special case player names
    if player_name in ["killer", "Playzone", "unknown"]:
        return player_name

    # Set similarity thresholds
    HIGH_SIMILARITY_THRESHOLD = 0.90  # 90% or higher is a definite match
    LOW_SIMILARITY_THRESHOLD = 0.5   # 60% is minimum for consideration

    player_name_lower = player_name.lower()
    
    # Collect all player names
    all_players = []
    for team, team_data in team_rosters.items():
        player_values = [team_data[f'player{i}'] for i in range(1, 9) if f'player{i}' in team_data and team_data[f'player{i}']]
        all_players.extend([(p, team) for p in player_values if p])

    # First pass: check for high-confidence matches (90%+)
    for player, team in all_players:
        similarity = fuzz.ratio(player_name_lower, player.lower()) / 100.0
        
        # If very close match found (≥90%), return immediately
        if similarity >= HIGH_SIMILARITY_THRESHOLD:
            return player
    
    # Second pass: find all matches above 60% threshold
    matches = []
    for player, team in all_players:
        similarity = fuzz.ratio(player_name_lower, player.lower()) / 100.0
        if similarity >= LOW_SIMILARITY_THRESHOLD:
            matches.append((player, team, similarity))
    
    # Sort matches by similarity score (highest first)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    # If we have matches above the threshold, return the best one
    if matches:
        best_player, best_team, best_score = matches[0]
        return best_player
    
    # Use RapidFuzz's process.extractOne as a fallback method
    if all_players:
        best_match_process = process.extractOne(
            player_name_lower, 
            [p[0].lower() for p in all_players],
            scorer=fuzz.ratio
        )
        
        if best_match_process and best_match_process[1] >= LOW_SIMILARITY_THRESHOLD * 100:
            best_idx = best_match_process[2]
            best_player_name = all_players[best_idx][0]
            return best_player_name
    
    # No good match found
    return "unknown"

def text_detection(kill_block_region, ocr):
    """
    Detect text in a kill block image.
    
    Args:
        kill_block_region (numpy.ndarray): Image region to detect text in
        ocr (PaddleOCR): OCR model instance
        
    Returns:
        list: List of detected text boxes with coordinates and text
    """
    try:
        # Prefixes to clean from player names
        prefixes_to_remove = []
        
        result = ocr.ocr(kill_block_region, cls=True)
        if not result:
            print(f"[OCR DEBUG] No OCR result for kill block")
            return []
            
        final = []
        for line in result:
            if line is None:
                continue
                
            for word_info in line:
                if not word_info or len(word_info) < 2:
                    continue
                    
                try:
                    text, confidence = word_info[1][0], word_info[1][1]

                    # Lowered confidence threshold from 0.5 to 0.3
                    # Lowered text length requirement from > 3 to > 1
                    if confidence > 0.3 and len(text) > 1:
                        print(f"[OCR DEBUG] Detected text: '{text}' with confidence: {confidence}")
                        
                        # Clean up text if needed
                        if ' ' in text and not text.endswith(' '):
                            text = text.split(' ')[1]
                            
                        # Get coordinates
                        x1, y1 = map(int, word_info[0][0])
                        x2, y2 = map(int, word_info[0][2])
                        
                        # Remove common prefixes
                        clean_text = text.strip()
                        for prefix in prefixes_to_remove:
                            if clean_text.startswith(prefix):
                                clean_text = clean_text[len(prefix):]
                        clean_text = clean_text.strip()
                        
                        if clean_text:
                            final.append((x1, y1, x2, y2, clean_text))
                            print(f"[OCR DEBUG] Added clean text: '{clean_text}'")
                except Exception as e:
                    print(f"[OCR DEBUG] Error processing word: {e}")
                    continue
                    
        print(f"[OCR DEBUG] Total text boxes detected: {len(final)}")
        return final
    except Exception as e:
        print(f"[OCR DEBUG] OCR exception: {e}")
        return []

def extract_kill_info(text_boxes, other_objects, x1, y1, x2, y2):
    """
    Extract killer, victim and weapon info from detected text and objects.
    
    Args:
        text_boxes (list): List of detected text boxes
        other_objects (list): List of other detected objects
        x1, y1, x2, y2 (int): Coordinates of the kill block
        
    Returns:
        tuple: (killer_name, victim_name, weapon_used)
    """
    killer_name = "unknown"
    victim_name = "unknown"
    weapon_used = "unknown"
    
    print(f"[EXTRACT DEBUG] Text boxes count: {len(text_boxes)}")
    print(f"[EXTRACT DEBUG] Text boxes: {text_boxes}")
    print(f"[EXTRACT DEBUG] Other objects count: {len(other_objects)}")
    print(f"[EXTRACT DEBUG] Other objects: {other_objects}")
    
    # Detect weapons inside the kill block
    detected_weapons = []
    for obj_coords in other_objects:
        ox1, oy1, ox2, oy2, label = obj_coords
        if is_inside((ox1, oy1, ox2, oy2), (x1, y1, x2, y2)):
            detected_weapons.append(label)
    
    print(f"[EXTRACT DEBUG] Detected weapons inside kill block: {detected_weapons}")
    
    # Determine weapon used from detected weapons
    if detected_weapons:
        if len(detected_weapons) > 1:
            # Check for special weapon types
            has_kill = any("kill" in w.lower() for w in detected_weapons)
            has_knockout = any("knockout" in w.lower() for w in detected_weapons)
            
            if has_kill and has_knockout:
                # Prefer knockout over kill
                weapon_used = next(w for w in detected_weapons if "knockout" in w.lower())
            elif has_kill:
                # Use first kill weapon
                weapon_used = next(w for w in detected_weapons if "kill" in w.lower())
        else:
            # Only one weapon detected
            weapon_used = detected_weapons[0]
    
    print(f"[EXTRACT DEBUG] Final weapon used: {weapon_used}")
    
    # Extract player names based on special cases
    if len(text_boxes) == 1:
        print(f"[EXTRACT DEBUG] Single text box case")
        if weapon_used == 'self-knockout':
            # Self-knockout case
            killer_name = text_boxes[0][4]
            victim_name = text_boxes[0][4]
            print(f"[EXTRACT DEBUG] Self-knockout: {killer_name}")
        elif weapon_used == 'kill':
            # Generic kill with single player name
            killer_name = "killer"
            victim_name = text_boxes[0][4]
            print(f"[EXTRACT DEBUG] Generic kill: killer -> {victim_name}")
        else:
            # Single text box but not self-knockout or generic kill
            # This might be a victim name only
            victim_name = text_boxes[0][4]
            print(f"[EXTRACT DEBUG] Single text box (victim only): {victim_name}")
    elif len(text_boxes) > 1:
        # Normal kill with killer and victim
        killer_name = text_boxes[0][4]
        victim_name = text_boxes[1][4]
        print(f"[EXTRACT DEBUG] Normal kill: {killer_name} -> {victim_name}")
    else:
        print(f"[EXTRACT DEBUG] No text boxes found")
    
    print(f"[EXTRACT DEBUG] Final result: Killer={killer_name}, Victim={victim_name}, Weapon={weapon_used}")
    return killer_name, victim_name, weapon_used

def is_inside(inner_box, outer_box):
    """
    Check if one bounding box is inside another.
    
    Args:
        inner_box (tuple): (x1, y1, x2, y2) of inner box
        outer_box (tuple): (x1, y1, x2, y2) of outer box
        
    Returns:
        bool: True if inner_box is inside outer_box
    """
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    
    # Check if boundaries are close enough
    upper_measurement = abs(oy1 - iy1)
    lower_measurement = abs(oy2 - iy2)
    
    if upper_measurement < 20 and lower_measurement < 20:
        return True
        
    # Traditional containment check
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2

def image_to_base64(image_path):
    """
    Convert an image file to base64 encoding.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64-encoded image or None if error
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

def save_to_mongodb(killer_name, weapon_used, victim_name, image_path, sift_weapon, position, collection, match_id=1, access_token=None, frame_timestamp=None):
    """
    Save kill event record to API instead of MongoDB and delete the local image.
    Prevents posting duplicate killfeeds by checking the last 5 events.
    Only killer_name, victim_name, and weapon_used (from API) are used for duplicate detection.
    """
    try:
        # Create a unique key for the event (exclude sift_weapon)
        event_key = f"{killer_name}|{victim_name}|{weapon_used}"
        
        # Only check for exact duplicates (all three values must match)
        if event_key in recent_killfeeds:
            # Clean up image even if duplicate
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                except:
                    pass
            return position
        
        # Add to recent queue
        recent_killfeeds.append(event_key)
        
        # Convert image to base64
        base64_image = image_to_base64(image_path)
        if not base64_image:
            return position
        
        # Create payload for API
        api_payload = {
            "killerName": killer_name,
            "victimName": victim_name,
            "weaponUsed": weapon_used,
            "imagePath": image_path,  
            "siftWeapon": sift_weapon[1],
            "image": base64_image
        }
        
        # Add timestamp for proper ordering if provided
        if frame_timestamp:
            api_payload["timestamp"] = frame_timestamp
        
        # Set headers with authorization token if provided
        headers = {
            'accept': 'text/plain',
            'Content-Type': 'application/json'
        }
        if access_token:
            headers['Authorization'] = f'Bearer {access_token}'
        
        # Make API request
        api_url = f'http://192.168.29.47:81/LeagueMatchData/LeagueMatch/LeagueMatchId/killfeed?matchId={match_id}'
        
        response = requests.post(
            api_url,
            headers=headers,
            json=api_payload
        )
        
        # Check if request was successful
        if response.status_code == 200 or response.status_code == 201:
            # Delete the image file after successful save
            try:
                os.remove(image_path)
            except Exception as e:
                pass
            return position + 1
        else:
            return position
    except Exception as e:
        pass
    finally:
        # Ensure cleanup even if errors occur
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass
    return position

def verify_weapon_with_sift(kill_block_folder, kill_type_from_api):
    """
    Verify weapon detection using SIFT matching against single-template images only.
    Args:
        kill_block_folder (str): Folder containing kill block images
        kill_type_from_api (str): Kill/knockout type from the API (e.g., kill, knockout, head-kill, etc.)
    Returns:
        tuple: (verification_result, detected_weapon_label)
    """
    # If weapon_used is a car or grenade kill, trust API and return
    api_weapons = [
        'molotov-kill', 'molotov-knockout', 'carblast-knockout', 'car-kill', 'grenade-knockout', 'grenade-kill','playzone-kill','playzone-knockout','self-knockout','self-kill','kill','carblast-kill','car-knockout','knife-kill','knife-knockout','knife-head-kill','knife-head-knockout'
    ]
    if kill_type_from_api in api_weapons:
        return True, kill_type_from_api

    try:
        sift = cv2.SIFT_create(nfeatures=400)
        best_weapon = None
        best_score = 0
        try:
            latest_kill_block = max(glob.glob(os.path.join(kill_block_folder, '*.jpg')), key=os.path.getctime)
        except Exception as e:
            return False, "unknown"
        target_image = cv2.imread(latest_kill_block)
        if target_image is None:
            return False, "unknown"
        gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.resize(gray_target, (0, 0), fx=0.75, fy=0.75)
        keypoints2, descriptors2 = sift.detectAndCompute(gray_target, None)
        if descriptors2 is None:
            return False, "unknown"
        for template_name, template_list in sift_template_cache.items():
            for descriptors1, template_image_path in template_list:
                try:
                    FLANN_INDEX_KDTREE = 1
                    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    if descriptors1.dtype != descriptors2.dtype:
                        descriptors1 = descriptors1.astype(np.float32)
                        descriptors2 = descriptors2.astype(np.float32)
                    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
                    if not matches:
                        continue
                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
                    if len(good_matches) > best_score:
                        best_score = len(good_matches)
                        best_weapon = template_name
                except Exception as e:
                    continue
        if best_weapon:
            final_weapon = f"{best_weapon}-{kill_type_from_api}"
            return True, final_weapon
        else:
            return False, "unknown"
    except Exception as e:
        return False, "unknown"

if __name__ == "__main__":
    obs_frame_capture()