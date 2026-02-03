import os
import warnings
import tensorflow as tf
import datetime
import time
from moviepy.editor import *
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from scipy.io import wavfile
import subprocess
import math
import cv2
import numpy as np
import psutil

# Configure warnings and logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# CONSTANTS AND CONFIGURATION
INPUT_FILES = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_FOLDER = os.path.join(BASE_DIR, "tmp")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "output")

# Create directories if they don't exist
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_FILE_NAME = "output"
SAMPLE_RATE = 24
THRESHOLD = 5
EXCEEDS_BY = 4
NO_OVERLAP_AUDIO = False
EMOTION_THRESHOLD = 0.5
CHECK_EMOTION_INTERVAL = 10

# Error handling constants
MAX_RETRIES = 3
TIMEOUT_SECONDS = 30
MIN_FRAME_SIZE = (64, 64)
MAX_FRAME_SIZE = (1920, 1080)

# GUI constants
ul_x = 10
ul_y = 10

def log_error(message, error):
    """Log errors to a file"""
    error_log_path = os.path.join(OUTPUT_FOLDER, "error_log.txt")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(error_log_path, "a", encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}: {str(error)}\n")

class EmotionDetector:
    def __init__(self):
        try:
            # Initialize FER detector
            from fer import FER
            self.emotion_detector = FER(mtcnn=True)  # Using MTCNN for better accuracy
            self.emotions_history = []
            self.last_detection_time = None
            self.detection_cooldown = 5  # 5 seconds cooldown
            print("Emotion detector initialized successfully")
        except Exception as e:
            log_error("Failed to initialize emotion detector", e)
            raise

    def detect_emotions_in_frame(self, frame):
        if frame is None or frame.size == 0:
            return None
            
        try:
            current_time = time.time()
            
            # Detect emotions using FER
            emotions = self.emotion_detector.detect_emotions(frame)
            
            if emotions:  # If any faces with emotions were detected
                dominant_emotion = None
                max_confidence = 0
                
                for face in emotions:
                    # Get the emotion with highest confidence
                    emotion_scores = face['emotions']
                    current_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                    
                    if current_emotion[1] > max_confidence:
                        max_confidence = current_emotion[1]
                        dominant_emotion = current_emotion[0]
                
                # Print emotion detection if cooldown has passed
                if (self.last_detection_time is None or 
                    current_time - self.last_detection_time >= self.detection_cooldown):
                    print(f"\nEmotion detected: {dominant_emotion} (confidence: {max_confidence:.2f})")
                    self.last_detection_time = current_time
                
                return (dominant_emotion, max_confidence)
            
            return None
            
        except Exception as e:
            log_error("Error detecting emotions in frame", e)
            return None

    def analyze_video_emotions(self, video_path):
        emotion_timestamps = []
        cap = None
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"\nAnalyzing emotions in {os.path.basename(video_path)}...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % CHECK_EMOTION_INTERVAL == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.detect_emotions_in_frame(frame_rgb)
                    
                    if result:
                        timestamp = frame_count / fps
                        emotion_timestamps.append({
                            'timestamp': timestamp,
                            'emotion': result[0],
                            'confidence': result[1]
                        })
                
                frame_count += 1
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"\rProgress: {progress:.1f}%", end='')
            
        except Exception as e:
            log_error(f"Error analyzing video", e)
        finally:
            if cap is not None:
                cap.release()
        
        return emotion_timestamps
    
class Window(Frame):
    def __init__(self, master=None):
        try:
            Frame.__init__(self, master)
            self.master = master
            self.check_system_requirements()
            
            try:
                self.emotion_detector = EmotionDetector()
            except Exception as e:
                print(f"Error initializing face detector: {e}")
                messagebox.showerror("Error", "Failed to initialize face detector")
                self.emotion_detector = None
                
            self.init_window()
            
        except Exception as e:
            print(f"Error initializing window: {e}")
            raise
    def check_system_requirements(self):
        try:
            min_ram = 8 * 1024 * 1024 * 1024  # 8GB in bytes
            available_ram = psutil.virtual_memory().available
            if available_ram < min_ram:
                messagebox.showwarning(
                    "System Requirements", 
                    f"Warning: Available RAM ({available_ram / 1024**3:.1f}GB) is less than recommended (8GB)"
                )
        except Exception as e:
            log_error("Error checking system requirements", e)

    def init_window(self):
        try:
            self.master.title("AutoPodcastEditor")
            self.pack(fill=BOTH, expand=1)
            
            # Main notice
            sync_notice = Label(self, text="Please ensure all input clips are in sync at start!")
            sync_notice.place(x=ul_x, y=ul_y)
            
            # File list frame
            self.file_list_frame = Frame(self)
            self.file_list_frame.place(x=ul_x, y=ul_y+50)
            
            # Add file button
            browseFileDir = Button(self, text="Add File", command=self.addFile)
            browseFileDir.place(x=ul_x, y=ul_y+25)

            # Settings frame
            settings_frame = LabelFrame(self, text="Settings", padx=5, pady=5)
            settings_frame.place(x=ul_x, y=ul_y + 260)

            # Sample Rate
            sampleRateLabel = Label(settings_frame, text="Sample Rate")
            sampleRateLabel.grid(row=0, column=0, padx=5)
            self.sampleRateEntry = Entry(settings_frame, width=3)
            self.sampleRateEntry.grid(row=0, column=1)
            self.sampleRateEntry.insert(END, str(SAMPLE_RATE))

            # Threshold
            thresholdLabel = Label(settings_frame, text="Threshold")
            thresholdLabel.grid(row=0, column=2, padx=5)
            self.thresholdEntry = Entry(settings_frame, width=3)
            self.thresholdEntry.grid(row=0, column=3)
            self.thresholdEntry.insert(END, str(THRESHOLD))

            # Exceeds By
            exceedsLabel = Label(settings_frame, text="Exceeds By")
            exceedsLabel.grid(row=0, column=4, padx=5)
            self.exceedsEntry = Entry(settings_frame, width=3)
            self.exceedsEntry.grid(row=0, column=5)
            self.exceedsEntry.insert(END, str(EXCEEDS_BY))

            # Face Detection Threshold
            emotionLabel = Label(settings_frame, text="Face Threshold")
            emotionLabel.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
            self.emotionEntry = Entry(settings_frame, width=3)
            self.emotionEntry.grid(row=1, column=2)
            self.emotionEntry.insert(END, str(EMOTION_THRESHOLD))

            # Overlap Audio
            self.overlap_var = BooleanVar(value=True)
            overlapAudioBox = Checkbutton(settings_frame, text="Overlap Audio", 
                                        variable=self.overlap_var, 
                                        command=self.toggleAudio)
            overlapAudioBox.grid(row=1, column=3, columnspan=3)

            # Output Name
            outputFrame = Frame(self)
            outputFrame.place(x=ul_x, y=ul_y + 345)
            
            outputNameLabel = Label(outputFrame, text="Output File Name")
            outputNameLabel.pack(side=LEFT)
            self.outputNameEntry = Entry(outputFrame, width=57)
            self.outputNameEntry.pack(side=LEFT, padx=5)
            self.outputNameEntry.insert(END, OUTPUT_FILE_NAME)

            # Process Button
            processButton = Button(self, text="Process", command=self.confirmSettings, 
                                 width=15, height=3)
            processButton.place(x=ul_x + 460, y=ul_y + 310)
            
        except Exception as e:
            log_error("Error initializing window components", e)
            raise

    def toggleAudio(self):
        global NO_OVERLAP_AUDIO
        NO_OVERLAP_AUDIO = not self.overlap_var.get()

    def addFile(self):
        try:
            filenames = askopenfilename(
                multiple=True,
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                    ("All files", "*.*")
                ]
            )
            
            for filename in filenames:
                if filename and filename not in INPUT_FILES:
                    INPUT_FILES.append(filename)
                    fileFrame = Frame(self.file_list_frame)
                    fileFrame.pack(anchor='w', pady=2)
                    
                    fileLabel = Label(fileFrame, text=os.path.basename(filename),
                                    foreground="blue")
                    fileLabel.pack(side=LEFT)
                    
                    removeBtn = Button(fileFrame, text="×", command=lambda f=filename, 
                                     fr=fileFrame: self.removeFile(f, fr))
                    removeBtn.pack(side=LEFT, padx=5)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error adding file(s): {e}")

    def removeFile(self, filename, frame):
        try:
            INPUT_FILES.remove(filename)
            frame.destroy()
        except Exception as e:
            log_error(f"Error removing file: {filename}", e)

    def confirmSettings(self):
        try:
            # Validate input files
            if not INPUT_FILES:
                messagebox.showerror("Error", "Please add at least one input file")
                return

            # Validate numeric inputs
            try:
                global SAMPLE_RATE, THRESHOLD, EXCEEDS_BY, OUTPUT_FILE_NAME, EMOTION_THRESHOLD
                SAMPLE_RATE = int(self.sampleRateEntry.get())
                THRESHOLD = int(self.thresholdEntry.get())
                EXCEEDS_BY = float(self.exceedsEntry.get())
                EMOTION_THRESHOLD = float(self.emotionEntry.get())
                OUTPUT_FILE_NAME = self.outputNameEntry.get()

                # Basic validation
                if SAMPLE_RATE <= 0 or THRESHOLD <= 0 or EXCEEDS_BY <= 0:
                    raise ValueError("Values must be positive")
                if EMOTION_THRESHOLD < 0 or EMOTION_THRESHOLD > 1:
                    raise ValueError("Face threshold must be between 0 and 1")
                    
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid settings: {str(e)}")
                return

            # Confirm processing
            if messagebox.askyesno("Confirm", "Start processing videos?"):
                self.spliceClips()
                
        except Exception as e:
            messagebox.showerror("Error", f"Error confirming settings: {e}")
    def parseAudioData(self, audioRate, audioArray):
        try:
            sampleDivider = math.floor(audioRate / SAMPLE_RATE)
            outputArray = []
            sampleCounter = 0
            
            while sampleCounter < audioArray.shape[0]:
                outputArray.append(audioArray[sampleCounter][0])
                sampleCounter += sampleDivider
                
            return outputArray
        except Exception as e:
            log_error("Error parsing audio data", e)
            raise

    def compareAudioArrays(self, audioArrays):
        try:
            priorityArray = 0
            consecutiveArray = 0
            prevArray = 0
            consecutiveCount = 0
            counter = 0
            outputArray = []

            audioArrays = self.normalizeArrays(audioArrays)

            while counter < len(audioArrays[0]):
                consecutiveArray = self.returnHighestIndex(audioArrays, counter, priorityArray)
                if consecutiveArray != prevArray:
                    prevArray = consecutiveArray
                    consecutiveCount = 1
                else:
                    consecutiveCount += 1
                    
                if consecutiveCount >= THRESHOLD:
                    priorityArray = consecutiveArray
                    
                outputArray.append(priorityArray)
                counter += 1

            return outputArray
            
        except Exception as e:
            log_error("Error comparing audio arrays", e)
            raise

    def returnHighestIndex(self, audioArrays, index, currentPriority):
        try:
            maxVal = 0
            returnIndex = 0
            
            for c in range(len(audioArrays)):
                current_value = abs(audioArrays[c][index])
                if c != currentPriority:
                    if current_value > maxVal:
                        maxVal = current_value
                        returnIndex = c
                else:
                    if current_value * EXCEEDS_BY > maxVal:
                        maxVal = current_value * EXCEEDS_BY
                        returnIndex = c
                        
            return returnIndex
            
        except Exception as e:
            log_error("Error finding highest index", e)
            raise

    def normalizeArrays(self, audioArrays):
        try:
            if not audioArrays:
                raise ValueError("No audio arrays provided")
                
            maxArrayLen = max(len(array) for array in audioArrays)
            outputArray = []
            
            for array in audioArrays:
                normalized_array = list(array)
                while len(normalized_array) < maxArrayLen:
                    normalized_array.append(0)
                outputArray.append(normalized_array)
                
            return outputArray
            
        except Exception as e:
            log_error("Error normalizing arrays", e)
            raise

    def is_quiet_moment(self, audioArrays, time_idx):
        try:
            threshold = 0.2
            if time_idx >= len(audioArrays[0]):
                return False
            return all(abs(array[time_idx]) < threshold for array in audioArrays)
        except Exception as e:
            log_error("Error checking quiet moment", e)
            return False

    def incorporate_emotions(self, outputArray, emotion_data, audioArrays, sample_rate):
        try:
            for time_idx in range(len(outputArray)):
                current_time = time_idx / sample_rate
                
                for video_idx, emotions in enumerate(emotion_data):
                    relevant_emotions = [e for e in emotions 
                                      if abs(e['timestamp'] - current_time) < 0.5]
                    
                    if relevant_emotions:
                        strongest_emotion = max(relevant_emotions, 
                                             key=lambda x: x['confidence'])
                        if (strongest_emotion['emotion'] == 'active'
                            and strongest_emotion['confidence'] > EMOTION_THRESHOLD):
                            if self.is_quiet_moment(audioArrays, time_idx):
                                outputArray[time_idx] = video_idx
                                
        except Exception as e:
            log_error("Error incorporating face detection", e)

    def create_progress_window(self):
        """
        Create progress window with time estimation
        """
        progress_window = Toplevel(self.master)
        progress_window.title("Processing Progress")
        progress_window.geometry("400x250")  # Made taller for new time estimate
        
        # Make window appear on top
        progress_window.transient(self.master)
        progress_window.grab_set()
        
        # Main progress label
        main_label = Label(progress_window, 
                        text="Processing Videos", 
                        font=('Arial', 14, 'bold'))
        main_label.pack(pady=10)
        
        # Current task label
        self.task_var = StringVar()
        task_label = Label(progress_window, 
                        textvariable=self.task_var, 
                        font=('Arial', 11))
        task_label.pack(pady=5)
        
        # Time estimate label
        self.time_var = StringVar()
        self.time_var.set("Calculating time remaining...")
        time_label = Label(progress_window, 
                        textvariable=self.time_var,
                        font=('Arial', 11))
        time_label.pack(pady=5)
        
        # Progress percentage
        self.progress_var = StringVar()
        progress_label = Label(progress_window, 
                            textvariable=self.progress_var,
                            font=('Arial', 11))
        progress_label.pack(pady=5)
        
        # Cancel button
        self.cancel_var = BooleanVar(value=False)
        cancel_btn = Button(progress_window, 
                        text="Cancel", 
                        command=lambda: self.cancel_var.set(True),
                        font=('Arial', 10))
        cancel_btn.pack(pady=10)
        
        # Initialize timing variables
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processed_items = 0
        self.total_items = 0
        
        return progress_window

    def update_progress(self, task, progress=None, total_items=None):
        """
        Update progress display with time estimation
        """
        current_time = time.time()
        
        # Initialize total items if provided
        if total_items is not None:
            self.total_items = total_items
            self.processed_items = 0
            self.start_time = current_time
            self.time_var.set("Calculating time remaining...")
            return

        # Update task
        self.task_var.set(task)
        
        if progress is not None:
            # Ensure progress is between 0 and 100
            validated_progress = max(0, min(100, float(progress)))
            self.progress_var.set(f"{validated_progress:.1f}%")
            
            # Update time estimate
            self.processed_items += 1
            if self.processed_items > 0 and self.total_items > 0:
                elapsed_time = current_time - self.start_time
                items_remaining = self.total_items - self.processed_items
                
                if self.processed_items > 1:  # Wait for at least 2 items for better estimate
                    avg_time_per_item = elapsed_time / self.processed_items
                    estimated_remaining = avg_time_per_item * items_remaining
                    
                    # Format time remaining
                    if estimated_remaining < 60:
                        time_str = f"About {int(estimated_remaining)} seconds remaining"
                    elif estimated_remaining < 3600:
                        minutes = int(estimated_remaining / 60)
                        seconds = int(estimated_remaining % 60)
                        time_str = f"About {minutes}m {seconds}s remaining"
                    else:
                        hours = int(estimated_remaining / 3600)
                        minutes = int((estimated_remaining % 3600) / 60)
                        time_str = f"About {hours}h {minutes}m remaining"
                    
                    # Add processing speed
                    items_per_second = self.processed_items / elapsed_time
                    time_str += f"\nProcessing speed: {items_per_second:.1f} segments/second"
                    
                    self.time_var.set(time_str)
                
        self.master.update()
    
    def spliceClips(self):
        progress_window = self.create_progress_window()
        try:
            # Calculate total steps for progress
            total_steps = (len(INPUT_FILES) * 2) + 3  # Audio + emotion analysis + final processing
            self.update_progress("Starting processing...", 0, total_steps)
            
            print("\nStarting video processing...")
            
            # Test video file access first
            for video_path in INPUT_FILES:
                if not os.path.exists(video_path):
                    raise ValueError(f"Video file not found: {video_path}")
                print(f"Found video file: {video_path}")
                
                # Test if video can be opened
                try:
                    test_clip = VideoFileClip(video_path)
                    print(f"Successfully opened {video_path}")
                    print(f"Duration: {test_clip.duration}, FPS: {test_clip.fps}")
                    test_clip.close()
                except Exception as e:
                    raise ValueError(f"Cannot open video {video_path}: {str(e)}")

            # Analyze emotions
            print("\nStarting emotion analysis...")
            emotion_data = []
            for i, video_path in enumerate(INPUT_FILES):
                if self.cancel_var.get():
                    raise Exception("Processing cancelled by user")
                    
                self.update_progress(f"Analyzing video {i+1}/{len(INPUT_FILES)}")
                emotions = self.emotion_detector.analyze_video_emotions(video_path)
                emotion_data.append(emotions)
                print(f"Completed emotion analysis for video {i+1}")

            # Process audio
            print("\nProcessing audio...")
            audioDataArrays = []
            for i, video_path in enumerate(INPUT_FILES):
                if self.cancel_var.get():
                    raise Exception("Processing cancelled by user")
                    
                self.update_progress(f"Processing audio {i+1}/{len(INPUT_FILES)}")
                try:
                    wav_path = os.path.join(TEMP_FOLDER, f"audio{i}.wav")
                    command = (
                        f'ffmpeg -i "{video_path}" -vn -acodec pcm_s16le '
                        f'-ar 44100 -ac 2 -y "{wav_path}"'
                    )
                    print(f"Extracting audio: {command}")
                    subprocess.call(command, shell=True)
                    
                    if not os.path.exists(wav_path):
                        raise ValueError(f"Failed to create WAV file: {wav_path}")
                        
                    audioRate, audioArray = wavfile.read(wav_path)
                    audioDataArrays.append(self.parseAudioData(audioRate, audioArray))
                    print(f"Successfully processed audio for video {i+1}")
                    
                except Exception as e:
                    log_error(f"Error processing audio for file {video_path}", e)
                    raise

            # Generate output array
            print("\nAnalyzing audio levels...")
            outputArray = self.compareAudioArrays(audioDataArrays)

            # Create final video
            print("\nCreating final video...")
            self.create_final_video(outputArray, emotion_data, progress_window)
            
            progress_window.destroy()
            messagebox.showinfo("Success", "Video processing complete!")
            
        except Exception as e:
            progress_window.destroy()
            if str(e) == "Processing cancelled by user":
                messagebox.showinfo("Cancelled", "Processing was cancelled")
            else:
                log_error("Error in splice clips", e)
                messagebox.showerror("Error", f"Error processing video: {str(e)}")
        finally:
            self.cleanup_temp_files()
    
    def create_final_video(self, outputArray, emotion_data, progress_window):
        try:
            outputClipList = []
            counter = 0
            prevPriority = -1
            prevEndPt = -1
            emotion_threshold = 0.7

            # Calculate switches
            switches = []
            prev = None
            for i in range(len(outputArray)):
                curr = outputArray[i]
                if curr != prev:
                    switches.append(i)
                    prev = curr
            total_switches = len(switches)
            
            # Update progress window
            self.task_var.set("Initializing video processing...")
            self.progress_var.set("0.0%")
            self.time_var.set("Calculating time remaining...")
            progress_window.update()
            
            print(f"\nStarting video creation process...")
            print(f"Total switches: {total_switches}")

            # Create emotion timeline for quick lookup
            emotion_timeline = {}
            print("\nProcessing emotion data...")
            for video_idx, video_emotions in enumerate(emotion_data):
                for emotion_info in video_emotions:
                    timestamp = emotion_info['timestamp']
                    confidence = emotion_info['confidence']
                    emotion = emotion_info['emotion']
                    
                    if confidence >= emotion_threshold:
                        time_key = int(timestamp * SAMPLE_RATE)
                        emotion_timeline[time_key] = {
                            'video_idx': video_idx,
                            'emotion': emotion,
                            'confidence': confidence
                        }
                        print(f"Strong emotion detected: {emotion} (confidence: {confidence:.2f}) "
                            f"at {timestamp:.2f}s from camera {video_idx}")
            
            # Initialize video clips dictionary
            video_clips = {}
            for idx, video_path in enumerate(INPUT_FILES):
                try:
                    self.task_var.set(f"Loading video {idx + 1} of {len(INPUT_FILES)}")
                    progress_window.update()
                    
                    clip = VideoFileClip(video_path)
                    video_clips[idx] = clip
                    print(f"Successfully loaded video {idx + 1}. Duration: {clip.duration}")
                except Exception as e:
                    print(f"Error loading video {idx}: {str(e)}")
                    raise

            current_switch = 0
            self.start_time = time.time()
            
            while counter < len(outputArray):
                if self.cancel_var.get():
                    raise Exception("Processing cancelled by user")
                    
                current_priority = None
                
                # Check for emotions at current time
                if counter in emotion_timeline:
                    emotion_info = emotion_timeline[counter]
                    current_priority = emotion_info['video_idx']
                    print(f"\nSwitching to camera {current_priority} due to "
                        f"{emotion_info['emotion']} (confidence: {emotion_info['confidence']:.2f})")
                else:
                    current_priority = outputArray[counter]

                # Handle first frame
                if prevEndPt == -1:
                    prevPriority = current_priority
                    prevEndPt = 0
                # Handle camera switch
                elif prevPriority != current_priority:
                    try:
                        # Update progress
                        current_switch += 1
                        progress = min(100, (current_switch / total_switches) * 100)
                        
                        # Update time estimate
                        current_time = time.time()
                        elapsed_time = current_time - self.start_time
                        if current_switch > 1:
                            items_remaining = total_switches - current_switch
                            avg_time_per_item = elapsed_time / current_switch
                            estimated_remaining = avg_time_per_item * items_remaining
                            
                            # Format time remaining
                            if estimated_remaining < 60:
                                time_str = f"About {int(estimated_remaining)} seconds remaining"
                            elif estimated_remaining < 3600:
                                minutes = int(estimated_remaining / 60)
                                seconds = int(estimated_remaining % 60)
                                time_str = f"About {minutes}m {seconds}s remaining"
                            else:
                                hours = int(estimated_remaining / 3600)
                                minutes = int((estimated_remaining % 3600) / 60)
                                time_str = f"About {hours}h {minutes}m remaining"
                            
                            self.time_var.set(time_str)
                        
                        self.task_var.set(f"Processing segment {current_switch}/{total_switches}")
                        self.progress_var.set(f"{progress:.1f}%")
                        progress_window.update()
                        
                        # Create video segment
                        main_clip = video_clips[prevPriority]
                        start_time = max(0, prevEndPt)
                        end_time = min(counter / SAMPLE_RATE, main_clip.duration)
                        
                        print(f"Creating segment from video {prevPriority}: "
                            f"{start_time:.2f}s to {end_time:.2f}s")
                        
                        subclip = main_clip.subclip(start_time, end_time)
                        
                        # Handle audio
                        if not NO_OVERLAP_AUDIO:
                            audio_clips = []
                            for idx, video in video_clips.items():
                                if video.audio is not None:
                                    audio_clip = video.audio.subclip(start_time, end_time)
                                    audio_clips.append(audio_clip)
                            
                            if audio_clips:
                                composite_audio = CompositeAudioClip(audio_clips)
                                subclip = subclip.set_audio(composite_audio)
                        
                        outputClipList.append(subclip)
                        
                    except Exception as e:
                        print(f"Error processing segment: {str(e)}")
                        
                    prevPriority = current_priority
                    prevEndPt = counter / SAMPLE_RATE
                
                counter += 1

            # Handle final segment
            if prevEndPt < (counter - 1) / SAMPLE_RATE:
                try:
                    self.task_var.set("Processing final segment")
                    progress_window.update()
                    
                    main_clip = video_clips[prevPriority]
                    end_time = min((counter - 1) / SAMPLE_RATE, main_clip.duration)
                    
                    print(f"\nCreating final segment from video {prevPriority}: "
                        f"{prevEndPt:.2f}s to {end_time:.2f}s")
                        
                    subclip = main_clip.subclip(prevEndPt, end_time)
                    
                    if not NO_OVERLAP_AUDIO:
                        audio_clips = []
                        for video in video_clips.values():
                            if video.audio is not None:
                                audio_clip = video.audio.subclip(prevEndPt, end_time)
                                audio_clips.append(audio_clip)
                        
                        if audio_clips:
                            composite_audio = CompositeAudioClip(audio_clips)
                            subclip = subclip.set_audio(composite_audio)
                            
                    outputClipList.append(subclip)
                    
                except Exception as e:
                    print(f"Error processing final segment: {str(e)}")

            # Create final video
            if not outputClipList:
                raise ValueError("No clips were created!")

            try:
                # Concatenate clips
                self.task_var.set("Concatenating video clips")
                self.progress_var.set("95%")
                progress_window.update()
                
                final_video = concatenate_videoclips(outputClipList, method="compose")
                
                # Write final video
                self.task_var.set("Writing final video file")
                self.progress_var.set("98%")
                progress_window.update()
                
                output_path = os.path.join(OUTPUT_FOLDER, f"{OUTPUT_FILE_NAME}.mp4")
                final_video.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile=os.path.join(TEMP_FOLDER, 'temp-audio.m4a'),
                    remove_temp=True,
                    audio=True,
                    threads=4,
                    preset='medium',
                    fps=24,
                    write_logfile=True,
                    verbose=False,
                    logger=None
                )
                
                # Show completion
                self.task_var.set("Processing complete!")
                self.progress_var.set("100%")
                self.time_var.set("Finished!")
                progress_window.update()
                
            except Exception as e:
                self.task_var.set("Error creating video!")
                self.time_var.set(str(e))
                progress_window.update()
                raise
                
            finally:
                # Cleanup
                try:
                    final_video.close()
                except:
                    pass
                    
                for clip in outputClipList:
                    try:
                        clip.close()
                    except:
                        pass
                        
                for clip in video_clips.values():
                    try:
                        clip.close()
                    except:
                        pass
                
        except Exception as e:
            log_error("Error creating final video", e)
            self.task_var.set("Error!")
            self.time_var.set(str(e))
            progress_window.update()
            raise
    
    def cleanup_temp_files(self):
        """Enhanced cleanup function that excludes important files"""
    try:
        print("\nCleaning up temporary files...")
        exclude_files = {"audio0.wav", "audio1.wav", "temp-audio.m4a.log"}
        for file in os.listdir(TEMP_FOLDER):
            file_path = os.path.join(TEMP_FOLDER, file)
            try:
                if os.path.isfile(file_path) and file not in exclude_files:
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
                else:
                    print(f"Skipping: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    except Exception as e:
        print(f"Error in cleanup: {e}")
def main():
    try:
        root = Tk()
        root.geometry("600x400")
        root.title("AutoPodcastEditor")
        
        # Set window icon if available
        try:
            root.iconbitmap("icon.ico")
        except:
            pass
        
        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                root.destroy()
                
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        app = Window(root)
        root.mainloop()
        
    except Exception as e:
        log_error("Critical error in main", e)
        messagebox.showerror("Critical Error", str(e))
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()