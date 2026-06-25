import cv2
import numpy as np
import os
import time
import subprocess
import sys
from urllib.parse import urlparse, parse_qs
import requests
import json
import tempfile
import shutil
import platform

def install_requirements():
    """Install required packages if not present"""
    packages = ['opencv-python', 'numpy', 'yt-dlp', 'requests']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    parsed = urlparse(url)
    if parsed.hostname in ['www.youtube.com', 'youtube.com', 'youtu.be']:
        if parsed.path == '/watch':
            return parse_qs(parsed.query)['v'][0]
        elif parsed.path.startswith('/live/'):
            return parsed.path.split('/live/')[1].split('?')[0]
        elif parsed.hostname == 'youtu.be':
            return parsed.path[1:]
    return None

def method_1_modern_yt_dlp(url):
    """Method 1: Modern yt-dlp with updated configurations"""
    try:
        print("Trying Method 1: Modern yt-dlp...")
        
        # Install yt-dlp if not available
        try:
            from yt_dlp import YoutubeDL
        except ImportError:
            print("Installing yt-dlp...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])
            from yt_dlp import YoutubeDL
        
        # Modern configurations that work with current YouTube
        configs = [
            {
                'format': 'best[height<=720]/best[height<=480]/worst',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'cookiesfrombrowser': ('chrome',),  # Use Chrome cookies if available
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            },
            {
                'format': 'worst[height<=360]/worst',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web'],
                    }
                }
            },
            {
                'format': 'worst',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            }
        ]
        
        for i, config in enumerate(configs):
            try:
                print(f"Trying config {i+1}...")
                with YoutubeDL(config) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    # Try to get direct URL
                    if 'url' in info:
                        print(f"Success with config {i+1}")
                        return info['url']
                    
                    # Try to get from formats
                    elif 'formats' in info and info['formats']:
                        # Sort formats by quality (prefer lower quality for streaming)
                        formats = sorted(info['formats'], key=lambda x: x.get('height', 0) or 0)
                        
                        for fmt in formats:
                            if fmt.get('url') and fmt.get('protocol') in ['http', 'https']:
                                print(f"Found format: {fmt.get('format_id')} ({fmt.get('height', 'N/A')}p)")
                                return fmt['url']
                        
                        # If no direct URLs, try to get the best available
                        for fmt in formats:
                            if fmt.get('url'):
                                print(f"Found format: {fmt.get('format_id')}")
                                return fmt['url']
                                
            except Exception as e:
                print(f"Config {i+1} failed: {e}")
                continue
        
        return None
    except Exception as e:
        print(f"Modern yt-dlp method failed: {e}")
        return None

def method_2_streamlink_modern(url):
    """Method 2: Modern Streamlink approach"""
    try:
        print("Trying Method 2: Modern Streamlink...")
        
        # Install streamlink if not available
        try:
            result = subprocess.run(['streamlink', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("Installing streamlink...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlink'])
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Installing streamlink...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlink'])
        
        # Try with modern streamlink options
        qualities = ['worst', '360p', '480p', '720p']
        
        for quality in qualities:
            try:
                print(f"Trying quality: {quality}")
                result = subprocess.run([
                    'streamlink', url, quality, '--stream-url',
                    '--user-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    stream_url = result.stdout.strip()
                    if stream_url:
                        print(f"Got stream URL for {quality}")
                        return stream_url
                        
            except subprocess.TimeoutExpired:
                continue
            except Exception as e:
                print(f"Streamlink {quality} failed: {e}")
                continue
        
        return None
    except Exception as e:
        print(f"Streamlink method failed: {e}")
        return None

def method_3_browser_simulation(url):
    """Method 3: Browser-like simulation"""
    try:
        print("Trying Method 3: Browser simulation...")
        
        # Try to simulate a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Try direct OpenCV with browser headers
        video_id = extract_video_id(url)
        if video_id:
            # Try different URL formats
            test_urls = [
                url,
                f"https://www.youtube.com/watch?v={video_id}",
                f"https://youtu.be/{video_id}"
            ]
            
            for test_url in test_urls:
                print(f"Testing URL: {test_url}")
                cap = cv2.VideoCapture(test_url)
                
                # Set additional properties to mimic browser
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print("Direct capture successful!")
                        cap.release()
                        return test_url
                    cap.release()
        
        return None
    except Exception as e:
        print(f"Browser simulation method failed: {e}")
        return None

def method_4_alternative_downloaders(url):
    """Method 4: Try alternative downloaders with live stream optimization"""
    try:
        print("Trying Method 4: Alternative downloaders...")
        
        # Try with yt-dlp using different extractors
        from yt_dlp import YoutubeDL
        
        # Multiple configs for different scenarios
        configs = [
            {
                'format': 'worst[height<=360]/worst',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['android'],
                        'player_skip': ['webpage'],
                    }
                }
            },
            {
                'format': 'worst[protocol!=m3u8]',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            },
            {
                'format': 'worst',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            }
        ]
        
        for i, config in enumerate(configs):
            try:
                print(f"Trying config {i+1}...")
                with YoutubeDL(config) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    if 'url' in info:
                        print(f"✅ Success with config {i+1}")
                        return info['url']
                    elif 'formats' in info and info['formats']:
                        # Get the worst quality format
                        formats = sorted(info['formats'], key=lambda x: x.get('height', 0) or 0)
                        for fmt in formats:
                            if fmt.get('url'):
                                print(f"✅ Found format with config {i+1}: {fmt.get('format_id')}")
                                return fmt['url']
            except Exception as e:
                print(f"Config {i+1} failed: {e}")
                continue
        
        return None
    except Exception as e:
        print(f"Alternative downloaders method failed: {e}")
        return None

def time_to_seconds(time_str):
    """Convert time string to seconds"""
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    elif len(parts) == 1:
        return parts[0]
    else:
        raise ValueError("Invalid time format")

def capture_frames_from_stream(stream_url, capture_interval=1.0, output_folder="captured_frames", 
                              start_sec=None, end_sec=None, max_duration=None):
    """Capture frames from a video stream with improved error handling"""
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Opening stream: {stream_url}")
    
    # Try multiple times to open the stream
    cap = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            cap = cv2.VideoCapture(stream_url)
            
            # Set additional properties for better streaming
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            if cap.isOpened():
                print(f"✅ Stream opened successfully on attempt {attempt + 1}")
                break
            else:
                print(f"❌ Failed to open stream on attempt {attempt + 1}")
                if cap:
                    cap.release()
                if attempt < max_retries - 1:
                    print("Retrying in 2 seconds...")
                    time.sleep(2)
        except Exception as e:
            print(f"❌ Error opening stream on attempt {attempt + 1}: {e}")
            if cap:
                cap.release()
            if attempt < max_retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
    
    if not cap or not cap.isOpened():
        print("❌ Failed to open stream after all attempts")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS
    
    print(f"Stream FPS: {fps}")
    
    frame_count = 0
    capture_count = 1
    start_time = time.time()
    last_progress = start_time
    consecutive_failures = 0
    max_consecutive_failures = 10
    
    # Calculate time-based capture intervals
    last_capture_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"❌ Failed to read frame (attempt {consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    print("❌ Too many consecutive failures, stopping capture")
                    break
                
                # Wait a bit before retrying
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful read
            consecutive_failures = 0
            
            current_time = time.time() - start_time
            
            # Progress update
            if time.time() - last_progress >= 2:
                print(f"Capturing... Time: {current_time:.1f}s, Frames: {frame_count}, Captured: {capture_count-1}")
                last_progress = time.time()
            
            # Check if we should stop based on duration
            if max_duration and current_time > max_duration:
                print(f"Reached maximum duration: {max_duration}s")
                break
            
            # Check if we should stop based on end time
            if end_sec and current_time > (end_sec - (start_sec or 0)):
                print(f"Reached end time")
                break
            
            # Skip frames before start time
            if start_sec and current_time < start_sec:
                frame_count += 1
                continue
            
            # Capture frame at time intervals (not frame intervals)
            if current_time - last_capture_time >= capture_interval:
                frame_path = os.path.join(output_folder, f'frame_{capture_count:04d}.jpg')
                
                # Verify frame is valid before saving
                if frame is not None and frame.size > 0:
                    success = cv2.imwrite(frame_path, frame)
                    if success:
                        print(f"✅ Captured frame {capture_count} at {current_time:.1f}s")
                        capture_count += 1
                        last_capture_time = current_time
                    else:
                        print(f"❌ Failed to save frame {capture_count}")
                else:
                    print(f"❌ Invalid frame, skipping capture")
            
            frame_count += 1
            
            # Safety check to prevent infinite loops
            if frame_count > 100000:  # About 55 minutes at 30fps
                print("Safety limit reached")
                break
                
    except KeyboardInterrupt:
        print("\nCapture interrupted by user")
    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        if cap:
            cap.release()
    
    print(f"\nCapture completed:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total frames captured: {capture_count-1}")
    print(f"Total duration: {time.time() - start_time:.1f}s")
    
    return capture_count > 1  # Return True if at least one frame was captured

def is_youtube_live(url):
    """Check if the YouTube URL is a live stream using yt-dlp."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])
        from yt_dlp import YoutubeDL
    ydl_opts = {'quiet': True, 'no_warnings': True}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('is_live', False)

def get_cookies_arg():
    """Return yt-dlp cookies argument if cookies.txt exists."""
    cookies_path = os.path.abspath("cookies.txt")
    if os.path.exists(cookies_path):
        print(f"Using cookies from {cookies_path}")
        return ['--cookies', cookies_path]
    return []

def download_youtube_segment(url, start_time, end_time, output_path, force_720p=False):
    """Download a segment of a YouTube video using yt-dlp, video only, at selected quality."""
    try:
        # Ensure yt-dlp is installed
        try:
            from yt_dlp import YoutubeDL
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])
        # Format time for yt-dlp
        section = f"*{start_time}-{end_time}"
        quality_str = "720p" if force_720p else "1080p+"
        print(f"Selected quality: {quality_str}")
        # Only download video (no audio) for smaller files
        format_str = 'bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]' if force_720p else 'bestvideo[height>=1080][ext=mp4]/best[height>=1080][ext=mp4]'
        cmd = [
            sys.executable, '-m', 'yt_dlp',
            '--download-sections', section,
            '-f', format_str,
            '-o', output_path,
        ] + get_cookies_arg() + [url]
        print(f"Downloading segment: {section} to {output_path} at {quality_str}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("yt-dlp failed:", result.stderr)
            if not force_720p:
                print("❌ 1080p+ segment not available or failed. Try 720p instead.")
            return False
        return True
    except Exception as e:
        print(f"Error downloading segment: {e}")
        return False

def download_youtube_full_vod(url, output_path):
    """Download the full YouTube VOD using yt-dlp."""
    try:
        try:
            from yt_dlp import YoutubeDL
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])
        cmd = [
            sys.executable, '-m', 'yt_dlp',
            '-f', 'best[ext=mp4]/best',
            '-o', output_path,
        ] + get_cookies_arg() + [url]
        print(f"Downloading full VOD to {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("yt-dlp full VOD failed:", result.stderr)
            return False
        return True
    except Exception as e:
        print(f"Error downloading full VOD: {e}")
        return False

def extract_frames_ffmpeg(video_path, output_folder, interval):
    """Extract frames from a video file using ffmpeg at the given interval (seconds)."""
    os.makedirs(output_folder, exist_ok=True)
    output_pattern = os.path.join(output_folder, 'frame_%04d.jpg')
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps=1/{interval}',
        '-q:v', '2',
        output_pattern
    ]
    print(f"Extracting frames using ffmpeg: {cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("ffmpeg failed:", result.stderr)
        return False
    print("Frame extraction complete.")
    return True

def automated_browser_record_and_extract():
    """Fully automated: browser playback, screen recording, and frame extraction."""
    import subprocess
    import time
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options

    # --- User input ---
    url = input("Enter YouTube video URL: ").strip()
    start_input = input("Enter start time (HH:MM:SS): ").strip()
    end_input = input("Enter end time (HH:MM:SS): ").strip()
    capture_interval = float(input("Enter frame capture interval in seconds (e.g. 0.5): ").strip())
    output_mp4 = "yt_segment.mp4"
    output_frames = "captured_frames"

    # --- Calculate duration ---
    def hms_to_seconds(hms):
        parts = [int(p) for p in hms.split(":")]
        if len(parts) == 3:
            return parts[0]*3600 + parts[1]*60 + parts[2]
        elif len(parts) == 2:
            return parts[0]*60 + parts[1]
        else:
            return int(parts[0])
    start_sec = hms_to_seconds(start_input)
    end_sec = hms_to_seconds(end_input)
    duration = end_sec - start_sec
    if duration <= 0:
        print("End time must be after start time.")
        return

    # --- 1. Start Chrome and Seek to Desired Time ---
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(10)  # Wait for page and player to load

    # Dismiss popups (if any)
    try:
        agree = driver.find_element(By.XPATH, "//button[contains(.,'Accept all')]")
        agree.click()
        time.sleep(2)
    except Exception:
        pass

    # --- Wait for ads to finish and skip if possible ---
    print("Checking for ads and waiting for them to finish (up to 60 seconds)...")
    ad_waited = 0
    max_ad_wait = 60
    while ad_waited < max_ad_wait:
        ad_playing = False
        # Try to click 'Skip Ads' if available
        try:
            skip_ad = driver.find_element(By.CLASS_NAME, 'ytp-ad-skip-button')
            print('Skip Ad button found, clicking...')
            skip_ad.click()
            time.sleep(1)
            ad_playing = True
        except Exception:
            pass
        # Check if ad overlay is present
        try:
            ad_overlay = driver.find_element(By.CLASS_NAME, 'ytp-ad-player-overlay')
            print('Ad is playing, waiting...')
            ad_playing = True
        except Exception:
            pass
        # Check if ad countdown is present
        try:
            ad_countdown = driver.find_element(By.CLASS_NAME, 'ytp-ad-duration-remaining')
            print('Ad countdown detected, waiting...')
            ad_playing = True
        except Exception:
            pass
        if not ad_playing:
            print('No ad detected, proceeding.')
            break
        time.sleep(1)
        ad_waited += 1
    else:
        print('Ad may still be playing after 60 seconds, proceeding anyway.')

    # Play the video if not already playing
    try:
        play_button = driver.find_element(By.CSS_SELECTOR, "button.ytp-play-button")
        if play_button.get_attribute("aria-label") == "Play (k)":
            play_button.click()
            time.sleep(1)
    except Exception:
        pass

    # Seek to the desired time
    driver.execute_script(f"""
        var player = document.querySelector('video');
        if (player) {{
            player.currentTime = {start_sec};
        }}
    """)
    time.sleep(2)

    # Go fullscreen for best quality
    try:
        fs_button = driver.find_element(By.CSS_SELECTOR, "button.ytp-fullscreen-button")
        fs_button.click()
        time.sleep(1)
    except Exception:
        pass

    # --- 2. Start ffmpeg Screen Recording ---
    print("Starting ffmpeg screen recording...")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "gdigrab",
        "-framerate", "30",
        "-i", "desktop",
        "-t", str(duration),
        "-vcodec", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        output_mp4
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd)
    print(f"Recording for {duration} seconds...")
    time.sleep(duration + 2)  # Wait for recording to finish

    # --- 3. Cleanup ---
    driver.quit()
    print(f"Recording saved as {output_mp4}")

    # --- 4. Extract frames from the MP4 ---
    print("Extracting frames from the recorded video...")
    extract_frames_ffmpeg(output_mp4, output_frames, capture_interval)
    print(f"Frames saved in {output_frames}/")

# --- HIGH QUALITY YouTube Frame Capture ---
def method_enhanced_yt_dlp_hq(url):
    """Enhanced yt-dlp method optimized for HIGH QUALITY (1080p+)"""
    try:
        print("🚀 Trying enhanced yt-dlp method (FORCE 1080p+ ONLY)...")
        from yt_dlp import YoutubeDL
        configs = [
            {
                'format': 'bestvideo[height>=1080]+bestaudio/best[height>=1080]',
                'quiet': False,
                'no_warnings': False,
                'extract_flat': False,
                'writesubtitles': False,
                'writeautomaticsub': False,
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web', 'android'],
                        'player_skip': ['webpage'],
                    }
                },
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Sec-Fetch-Mode': 'navigate',
                }
            }
        ]
        for i, config in enumerate(configs):
            try:
                print(f"📡 Trying HIGH QUALITY config {i+1}/{len(configs)}...")
                with YoutubeDL(config) as ydl:
                    info = ydl.extract_info(url, download=False)
                    print(f"📺 Video title: {info.get('title', 'Unknown')}")
                    print(f"🔴 Is live: {info.get('is_live', False)}")
                    print(f"⏱️ Duration: {info.get('duration', 'Unknown')}")
                    if 'formats' in info and info['formats']:
                        print("🎥 Available formats:")
                        for fmt in info['formats'][:10]:
                            height = fmt.get('height', 'N/A')
                            width = fmt.get('width', 'N/A')
                            fps = fmt.get('fps', 'N/A')
                            ext = fmt.get('ext', 'N/A')
                            format_id = fmt.get('format_id', 'N/A')
                            vcodec = fmt.get('vcodec', 'N/A')
                            print(f"   {format_id}: {width}x{height} @ {fps}fps, {ext}, {vcodec}")
                    if 'url' in info:
                        selected_format = info
                        print(f"✅ Selected format: {selected_format.get('width', 'N/A')}x{selected_format.get('height', 'N/A')} @ {selected_format.get('fps', 'N/A')}fps")
                        return info['url']
                    elif 'formats' in info and info['formats']:
                        valid_formats = []
                        for fmt in info['formats']:
                            if (fmt.get('url') and 
                                fmt.get('protocol') in ['http', 'https', 'hls'] and
                                fmt.get('vcodec') != 'none' and (fmt.get('height', 0) or 0) >= 1080):
                                valid_formats.append(fmt)
                        if valid_formats:
                            def quality_score(fmt):
                                height = fmt.get('height', 0) or 0
                                width = fmt.get('width', 0) or 0
                                fps = fmt.get('fps', 0) or 0
                                return height * width * fps
                            valid_formats.sort(key=quality_score, reverse=True)
                            for fmt in valid_formats:
                                height = fmt.get('height', 'N/A')
                                width = fmt.get('width', 'N/A')
                                fps = fmt.get('fps', 'N/A')
                                ext = fmt.get('ext', 'N/A')
                                format_id = fmt.get('format_id', 'N/A')
                                print(f"🎥 Using format: {format_id} ({width}x{height} @ {fps}fps) - {ext}")
                                return fmt['url']
                        else:
                            print("❌ No 1080p+ formats found!")
                            return None
            except Exception as e:
                print(f"❌ HIGH QUALITY Config {i+1} failed: {e}")
                continue
        return None
    except Exception as e:
        print(f"❌ Enhanced HIGH QUALITY yt-dlp method failed: {e}")
        return None

def download_and_capture_frames_hq(url, capture_interval=1.0, output_folder="captured_frames_hq", 
                                  start_sec=None, end_sec=None, max_duration=None, force_720p=False):
    """Download video segment and capture frames in HIGH QUALITY (1080p+ or 720p)"""
    try:
        quality_str = "720p" if force_720p else "1080p+"
        print(f"📥 Using HIGH QUALITY download and capture method (FORCE {quality_str} ONLY, SEGMENT if possible)...")
        from yt_dlp import YoutubeDL
        os.makedirs(output_folder, exist_ok=True)
        
        # Store video file in the output folder instead of temp
        video_filename = "downloaded_video.%(ext)s"
        video_path = os.path.join(output_folder, video_filename)
        
        ydl_opts = {
            'format': 'bestvideo[height<=720]+bestaudio/best[height<=720]' if force_720p else 'bestvideo[height>=1080]+bestaudio/best[height>=1080]',
            'outtmpl': video_path,
            'quiet': False,
            'no_warnings': False,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'merge_output_format': 'mp4',
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
        # If both start_sec and end_sec are provided, use download_sections
        if start_sec is not None and end_sec is not None and end_sec > start_sec:
            def sec_to_hms(sec):
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = int(sec % 60)
                return f"{h:02d}:{m:02d}:{s:02d}"
            start_hms = sec_to_hms(start_sec)
            end_hms = sec_to_hms(end_sec)
            ydl_opts['download_sections'] = f"*{start_hms}-{end_hms}"
            print(f"⏳ Will download segment: {start_hms} to {end_hms}")
        elif start_sec is not None or end_sec is not None:
            print("⚠️ Both start and end time must be provided for segment download. Downloading full video instead.")
        if max_duration:
            ydl_opts['external_downloader_args'] = ['-t', str(max_duration)]
        print(f"📡 Downloading HIGH QUALITY video at {quality_str}...")
        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                # Check if the best format is at least 720p or 1080p
                best_format = None
                min_height = 720 if force_720p else 1080
                if 'formats' in info:
                    for fmt in info['formats']:
                        if (fmt.get('height', 0) or 0) >= min_height and fmt.get('vcodec') != 'none':
                            best_format = fmt
                            break
                if not best_format:
                    print(f"❌ No {quality_str} video format available for this video!")
                    return False
                print(f"🎥 Selected format: {best_format.get('format_id', 'N/A')} {best_format.get('width', 'N/A')}x{best_format.get('height', 'N/A')} @ {best_format.get('fps', 'N/A')}fps")
                info = ydl.extract_info(url, download=True)
                downloaded_file = None
                for ext in ['mp4', 'webm', 'mkv', 'flv']:
                    test_path = video_path.replace('%(ext)s', ext)
                    if os.path.exists(test_path):
                        downloaded_file = test_path
                        break
                if not downloaded_file:
                    print("❌ Could not find downloaded file")
                    return False
                file_size = os.path.getsize(downloaded_file) / (1024*1024)
                print(f"✅ Downloaded {quality_str} video to: {downloaded_file}")
                print(f"📦 File size: {file_size:.1f} MB")
        except Exception as e:
            print(f"❌ {quality_str} Download failed: {e}")
            return False
        return capture_frames_from_file_hq(
            downloaded_file, 
            capture_interval, 
            output_folder,
            start_sec, 
            end_sec, 
            max_duration,
            force_720p=force_720p
        )
    except Exception as e:
        print(f"❌ {quality_str} Download and capture method failed: {e}")
        return False

def capture_frames_from_file_hq(video_path, capture_interval=1.0, output_folder="captured_frames_hq", 
                               start_sec=None, end_sec=None, max_duration=None, force_720p=False):
    """Capture frames from a local video file in HIGH QUALITY (1080p+ or 720p)"""
    try:
        print(f"🎬 Capturing HIGH QUALITY frames from: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Failed to open video file")
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        print(f"📊 HIGH QUALITY Video info:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {duration:.1f}s")
        # Abort if not the selected quality
        min_height = 720 if force_720p else 1080
        if height < min_height:
            print(f"❌ Video is not {min_height}p or higher (actual: {width}x{height}). Aborting frame extraction.")
            cap.release()
            try:
                os.unlink(video_path)
            except:
                pass
            return False
        frame_count = 0
        capture_count = 1
        last_capture_time = 0
        start_frame = int(start_sec * fps) if start_sec else 0
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_count = start_frame
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("✅ Reached end of video")
                    break
                current_time = frame_count / fps
                if max_duration and current_time > (start_sec or 0) + max_duration:
                    print(f"⏱️ Reached maximum duration: {max_duration}s")
                    break
                if end_sec and current_time > end_sec:
                    print(f"⏱️ Reached end time: {end_sec}s")
                    break
                if current_time - last_capture_time >= capture_interval:
                    frame_path = os.path.join(output_folder, f'frame_hq_{capture_count:04d}.jpg')
                    if frame is not None and frame.size > 0:
                        jpeg_quality = [cv2.IMWRITE_JPEG_QUALITY, 95]
                        success = cv2.imwrite(frame_path, frame, jpeg_quality)
                        if success:
                            file_size = os.path.getsize(frame_path) / 1024
                            print(f"📸 Captured HIGH QUALITY frame {capture_count} at {current_time:.1f}s ({file_size:.1f} KB)")
                            capture_count += 1
                            last_capture_time = current_time
                        else:
                            print(f"❌ Failed to save HIGH QUALITY frame {capture_count}")
                    else:
                        print(f"❌ Invalid frame, skipping capture")
                frame_count += 1
                if frame_count % (int(fps) * 5) == 0:
                    progress = (current_time / duration * 100) if duration > 0 else 0
                    print(f"⏳ Progress: {progress:.1f}% ({current_time:.1f}s/{duration:.1f}s)")
        except KeyboardInterrupt:
            print("\n⚠️ Capture interrupted by user")
        except Exception as e:
            print(f"❌ Error during HIGH QUALITY capture: {e}")
        finally:
            cap.release()
            print(f"💾 Video file kept in: {video_path}")
        print(f"\n📋 HIGH QUALITY Capture Summary:")
        print(f"   Video resolution: {width}x{height}")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Total HIGH QUALITY frames captured: {capture_count-1}")
        print(f"   Output folder: {output_folder}")
        if capture_count > 1:
            sample_frame = os.path.join(output_folder, 'frame_hq_0001.jpg')
            if os.path.exists(sample_frame):
                sample_size = os.path.getsize(sample_frame) / 1024
                print(f"   Sample frame size: {sample_size:.1f} KB")
        return capture_count > 1
    except Exception as e:
        print(f"❌ HIGH QUALITY Frame capture from file failed: {e}")
        return False

def main_hq(output_folder="captured_frames_hq"):
    print("YouTube HIGH QUALITY Frame Capture Tool")
    print("=" * 50)
    url = input("Enter YouTube video URL: ").strip()
    try:
        capture_interval = float(input("Enter capture interval in seconds (e.g. 0.5): ").strip())
        if capture_interval <= 0:
            raise ValueError("Interval must be positive")
    except ValueError as e:
        print(f"Invalid interval: {e}")
        return
    start_input = input("Enter start time (HH:MM:SS) or press Enter: ").strip()
    end_input = input("Enter end time (HH:MM:SS) or press Enter: ").strip()
    start_sec = None
    end_sec = None
    max_duration = None
    if start_input or end_input:
        try:
            if start_input:
                start_sec = time_to_seconds(start_input)
            if end_input:
                end_sec = time_to_seconds(end_input)
            if start_sec is not None and end_sec is not None:
                max_duration = end_sec - start_sec
                if max_duration <= 0:
                    raise ValueError("End time must be after start time")
        except Exception as e:
            print(f"Invalid time input: {e}")
            return
    print("Select quality:")
    print("1. 1080p+ (largest file, best quality)")
    print("2. 720p (smaller file, good quality)")
    qmode = input("Enter 1 or 2: ").strip()
    force_720p = (qmode == "2")
    quality_str = "720p" if force_720p else "1080p+"
    print(f"\n🚀 Starting {quality_str} frame capture...")
    print(f"📺 URL: {url}")
    print(f"⏱️ Interval: {capture_interval}s")
    if start_sec: print(f"🕐 Start: {start_sec}s")
    if end_sec: print(f"🕐 End: {end_sec}s")
    if max_duration: print(f"⌛ Duration: {max_duration}s")
    print(f"\n⚠️  {quality_str.upper()} MODE ENABLED")
    print("   📈 This will download the selected quality")
    print("   ⏳ Download may take longer for higher quality")
    print("   💾 File sizes will be larger for higher quality")
    success = download_and_capture_frames_hq(
        url, 
        capture_interval, 
        output_folder=output_folder,
        start_sec=start_sec, 
        end_sec=end_sec,
        max_duration=max_duration,
        force_720p=force_720p
    )
    if success:
        print(f"\n✅ {quality_str.upper()} Frame capture completed successfully!")
        print(f"📁 Check the '{output_folder}' folder for your images")
        print("🔍 Compare the quality with your previous captures")
    else:
        print(f"\n❌ {quality_str.upper()} Frame capture failed!")
        print("\n🔧 Troubleshooting suggestions:")
        print("1. The video might not have the selected quality available")
        print("2. Try a different time range")
        print("3. Check your internet connection for large downloads")

# --- END HIGH QUALITY ---

def main(output_folder="captured_frames"):
    print("YouTube Frame Capture Tool (Updated)")
    print("=" * 50)
    url = input("Enter YouTube video URL: ").strip()
    try:
        capture_interval = float(input("Enter capture interval in seconds (e.g. 0.5): ").strip())
        if capture_interval <= 0:
            raise ValueError("Interval must be positive")
    except ValueError as e:
        print(f"Invalid interval: {e}")
        return
    start_input = input("Enter start time (HH:MM:SS) or press Enter: ").strip()
    end_input = input("Enter end time (HH:MM:SS) or press Enter: ").strip()
    start_sec = None
    end_sec = None
    max_duration = None
    if start_input or end_input:
        try:
            if start_input:
                start_sec = time_to_seconds(start_input)
            if end_input:
                end_sec = time_to_seconds(end_input)
            if start_sec is not None and end_sec is not None:
                max_duration = end_sec - start_sec
                if max_duration <= 0:
                    raise ValueError("End time must be after start time")
        except Exception as e:
            print(f"Invalid time input: {e}")
            return
    print("Select quality:")
    print("1. 1080p+ (video only, best quality)")
    print("2. 720p (video only, smaller file)")
    qmode = input("Enter 1 or 2: ").strip()
    force_720p = (qmode == "2")
    quality_str = "720p" if force_720p else "1080p+"
    is_live = is_youtube_live(url)
    segment_path = os.path.abspath("yt_segment.mp4")
    vod_path = os.path.abspath("yt_full_vod.mp4")
    segment_success = False
    if not is_live and start_input and end_input:
        segment_success = download_youtube_segment(url, start_input, end_input, segment_path, force_720p=force_720p)
        if segment_success:
            print(f"Downloaded segment to {segment_path}. Extracting frames...")
            extract_frames_ffmpeg(segment_path, output_folder, capture_interval)
            print("Done!")
            return
        else:
            print(f"Failed to download the specified segment at {quality_str}. Try a different quality or time range.")
            return
    else:
        stream_url = None
        stream_url = method_1_modern_yt_dlp(url)
        if not stream_url:
            stream_url = method_2_streamlink_modern(url)
        if not stream_url:
            stream_url = method_3_browser_simulation(url)
        if not stream_url:
            stream_url = method_4_alternative_downloaders(url)
        if stream_url:
            print(f"\nStarting frame capture...")
            success = capture_frames_from_stream(
                stream_url, 
                capture_interval, 
                output_folder=output_folder,
                start_sec=start_sec, 
                end_sec=end_sec,
                max_duration=max_duration
            )
            if success:
                print("\nFrame capture completed successfully!")
            else:
                print("\nFrame capture failed!")
        else:
            print("\nFailed to access video stream with all methods!")
            print("\nTroubleshooting suggestions:")
            print("1. Make sure the YouTube URL is accessible and not private/restricted")
            print("2. Try using a different YouTube URL")
            print("3. Check if the video is region-restricted")
            print("4. Use browser extensions to download the video")
            print("5. Wait for the live stream to end and try with the recorded version")

# Patch the __main__ block to add option 3 for HD mode
if __name__ == "__main__":
    print("Select mode:")
    print("1. Legacy (yt-dlp/streamlink)")
    print("3. High Quality (HD) Download & Frame Capture")
    mode = input("Enter 1 or 3: ").strip()
    # Prompt for output folder name
    output_folder = input("Enter folder name to store frames (will be created if not exists): ").strip()
    import os
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = "captured_frames"
        os.makedirs(output_folder, exist_ok=True)
    if mode == "3":
        main_hq(output_folder=output_folder)
    else:
        main(output_folder=output_folder)