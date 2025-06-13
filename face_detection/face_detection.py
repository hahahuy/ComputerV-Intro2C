import cv2
import time
import numpy as np
from collections import deque
from concurrent.futures import ThreadPoolExecutor

class FaceDetector:
    def __init__(self):
        # Load multiple cascade classifiers for better accuracy
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Performance optimization - Dynamic frame skipping
        self.base_frame_skip = 2  # Minimum frame skip
        self.max_frame_skip = 8   # Maximum frame skip
        self.current_frame_skip = self.base_frame_skip
        self.frame_count = 0
        self.target_fps = 25.0    # Target FPS for dynamic adjustment
        
        # Multi-scale detection
        self.detection_scales = [0.3, 0.5, 0.7]
        
        # FPS calculation and dynamic adjustment
        self.fps_queue = deque(maxlen=10)  # Shorter queue for faster adaptation
        self.last_fps_check = time.time()
        self.fps_check_interval = 1.0  # Check FPS every second
        
        # Pre-allocated frame buffers
        self.frame_width = 640
        self.frame_height = 480
        self.gray_frame = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        self.small_frames = {}
        for scale in self.detection_scales:
            w, h = int(self.frame_width * scale), int(self.frame_height * scale)
            self.small_frames[scale] = np.zeros((h, w), dtype=np.uint8)
        
        # Background subtraction for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
        self.motion_threshold = 1000  # Minimum motion pixels to trigger detection
        self.bg_learning_rate = 0.05  # Default learning rate
        
        # Frame pooling for reduced garbage collection
        self.frame_pool = deque(maxlen=5)
        self.temp_frame_pool = deque(maxlen=3)
        
        # Creative enhancements
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        self.face_effects = ['normal', 'blur', 'pixelate']
        self.current_effect = 0
        
        # Store last detected faces for smoother display
        self.last_faces = []
        self.last_detection_time = time.time()
        
        # Add face position smoothing over multiple frames
        self.face_history = deque(maxlen=5)  # Store last 5 detections
        self.confidence_threshold = 0.6     # Only show confident detections
        
        # Multi-threading configuration
        self.use_multithreading = False  # Start with threading disabled for safety
        self.max_workers = min(2, len(self.detection_scales))  # Limit workers to 2 for safety
        self.thread_pool = None  # Initialize as None, create when needed

    def adjust_frame_skip_dynamically(self, current_fps):
        """Dynamically adjust frame skipping based on current FPS"""
        current_time = time.time()
        if current_time - self.last_fps_check >= self.fps_check_interval:
            if current_fps < self.target_fps * 0.8:  # If FPS is too low
                self.current_frame_skip = min(self.current_frame_skip + 1, self.max_frame_skip)
            elif current_fps > self.target_fps * 1.2:  # If FPS is too high
                self.current_frame_skip = max(self.current_frame_skip - 1, self.base_frame_skip)
            
            self.last_fps_check = current_time

    def get_pooled_frame(self, shape):
        """Get a frame from pool or create new one"""
        if self.temp_frame_pool:
            frame = self.temp_frame_pool.popleft()
            if frame.shape == shape:
                return frame
        return np.zeros(shape, dtype=np.uint8)

    def return_frame_to_pool(self, frame):
        """Return frame to pool for reuse"""
        if len(self.temp_frame_pool) < self.temp_frame_pool.maxlen:
            self.temp_frame_pool.append(frame)

    def detect_motion(self, gray_frame):
        """Detect motion using background subtraction"""
        fg_mask = self.bg_subtractor.apply(gray_frame, learningRate=self.bg_learning_rate)
        motion_pixels = cv2.countNonZero(fg_mask)
        return motion_pixels > self.motion_threshold

    def non_max_suppression(self, boxes, scores, score_threshold=0.3, nms_threshold=0.4):
        """Apply Non-Maximum Suppression to eliminate duplicate detections"""
        if len(boxes) == 0:
            return []
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes_list = []
        scores_list = []
        
        for i, (x, y, w, h) in enumerate(boxes):
            boxes_list.append([int(x), int(y), int(w), int(h)])
            scores_list.append(float(scores[i]))
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, score_threshold, nms_threshold)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [boxes[i] for i in indices]
        return []

    def detect_faces_multiscale(self, gray_frame):
        """Multi-scale face detection with NMS (sequential for stability)"""
        # Always use sequential processing for face detection to avoid OpenCV threading issues
        return self._detect_faces_sequential(gray_frame)

    def _detect_faces_sequential(self, gray_frame):
        """Sequential face detection (fallback method)"""
        all_faces = []
        all_scores = []
        
        for scale in self.detection_scales:
            # Use pre-allocated buffer
            small_frame = self.small_frames[scale]
            cv2.resize(gray_frame, (small_frame.shape[1], small_frame.shape[0]), dst=small_frame)
            
            # Detect faces at this scale
            faces = self.face_cascade.detectMultiScale(
                small_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale back to original size and add confidence scores
            for (x, y, w, h) in faces:
                scaled_x = int(x / scale)
                scaled_y = int(y / scale)
                scaled_w = int(w / scale)
                scaled_h = int(h / scale)
                
                all_faces.append((scaled_x, scaled_y, scaled_w, scaled_h))
                # Simple confidence based on face size (larger faces = higher confidence)
                confidence = min(1.0, (scaled_w * scaled_h) / (100 * 100))
                all_scores.append(confidence)
            
            # Try profile detection if no frontal faces at this scale
            if len(faces) == 0:
                profile_faces = self.profile_cascade.detectMultiScale(
                    small_frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                for (x, y, w, h) in profile_faces:
                    scaled_x = int(x / scale)
                    scaled_y = int(y / scale)
                    scaled_w = int(w / scale)
                    scaled_h = int(h / scale)
                    
                    all_faces.append((scaled_x, scaled_y, scaled_w, scaled_h))
                    confidence = min(0.8, (scaled_w * scaled_h) / (100 * 100))  # Slightly lower confidence for profile
                    all_scores.append(confidence)
        
        # Apply Non-Maximum Suppression
        if all_faces:
            filtered_faces = self.non_max_suppression(all_faces, all_scores)
            return filtered_faces
        
        return []

    def apply_face_effect(self, frame, face_rect, color, effect_type):
        """Apply creative effects to detected faces"""
        x, y, w, h = face_rect
        
        # Bounds checking to prevent array index errors
        if (y + h > frame.shape[0] or x + w > frame.shape[1] or 
            x < 0 or y < 0 or w <= 0 or h <= 0):
            return
        
        if effect_type == 'normal':
            # Simple rectangular outline without any face modification
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
        elif effect_type == 'blur':
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                blurred = cv2.GaussianBlur(face_roi, (51, 51), 0)
                frame[y:y+h, x:x+w] = blurred
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
        elif effect_type == 'pixelate':
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size > 0:
                # Pixelate effect
                small = cv2.resize(face_roi, (16, 16), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y:y+h, x:x+w] = pixelated
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    def detect_features(self, frame, face_rect):
        """Detect eyes and smile within face region using pre-converted grayscale"""
        x, y, w, h = face_rect
        
        # Bounds checking to prevent array index errors
        if (y + h > self.gray_frame.shape[0] or x + w > self.gray_frame.shape[1] or 
            x < 0 or y < 0 or w <= 0 or h <= 0):
            return 0, 0
            
        face_roi_gray = self.gray_frame[y:y+h, x:x+w]
        
        # Additional safety check for empty ROI
        if face_roi_gray.size == 0:
            return 0, 0
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5, minSize=(10, 10))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
        
        # Detect smile
        smiles = self.smile_cascade.detectMultiScale(face_roi_gray, 1.8, 20, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 255), 2)
            cv2.putText(frame, "SMILE!", (x+sx, y+sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return len(eyes), len(smiles)

    def add_ui_elements(self, frame, fps, face_count, motion_detected):
        """Add UI elements and information overlay"""
        height, width = frame.shape[:2]
        
        # Semi-transparent overlay for UI
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 170), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # FPS and face count
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {face_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Effect: {self.face_effects[self.current_effect]}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame Skip: {self.current_frame_skip}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Motion detection indicator
        motion_color = (0, 255, 0) if motion_detected else (0, 0, 255)
        cv2.putText(frame, f"Motion: {'YES' if motion_detected else 'NO'}", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
        
        # Threading status
        threading_status = "ON" if self.use_multithreading else "OFF"
        threading_color = (0, 255, 255) if self.use_multithreading else (128, 128, 128)
        cv2.putText(frame, f"Threading: {threading_status}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, threading_color, 2)
        
        cv2.putText(frame, "Press 'e' to change effect, 't' to toggle threading, 'q' to quit", (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def assess_face_quality(self, face_roi):
        # Check face sharpness, contrast, and size
        laplacian_var = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        contrast = face_roi.std()
        return laplacian_var > 100 and contrast > 30  # Quality thresholds

    def adaptive_quality_control(self, current_fps):
        """Dynamically adjust detection quality based on performance"""
        old_scales = self.detection_scales.copy()
        
        if current_fps < 15:
            self.detection_scales = [0.5]  # Single scale only
            self.current_frame_skip = 6
        elif current_fps < 20:
            self.detection_scales = [0.4, 0.6]  # Two scales
            self.current_frame_skip = 4
        else:
            self.detection_scales = [0.3, 0.5, 0.7]  # Full quality
            self.current_frame_skip = 2
        
        # Update small_frames buffers if scales changed
        if old_scales != self.detection_scales:
            self.small_frames = {}
            for scale in self.detection_scales:
                w, h = int(self.frame_width * scale), int(self.frame_height * scale)
                self.small_frames[scale] = np.zeros((h, w), dtype=np.uint8)

    def adaptive_background_learning(self, motion_level):
        """Adjust background learning rate based on motion level"""
        if motion_level < 500:  # Very little motion
            learning_rate = 0.01  # Slow learning
        elif motion_level > 5000:  # High motion
            learning_rate = 0.1   # Fast learning
        else:
            learning_rate = 0.05  # Normal learning
        
        # Store learning rate for next background subtraction call
        self.bg_learning_rate = learning_rate

    def enhance_frame_for_detection(self, gray_frame):
        # Histogram equalization for better detection in poor lighting
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(gray_frame)

    def detect_face_orientation(self, face_roi):
        # Use eye detection to determine face orientation
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        if len(eyes) >= 2:
            # Calculate angle between eyes
            eye1, eye2 = eyes[0], eyes[1]
            angle = np.arctan2(eye2[1] - eye1[1], eye2[0] - eye1[0])
            return np.degrees(angle)
        return 0

    def detect_faces_at_scale(self, gray_frame, scale):
        """Detect faces at a specific scale - designed for parallel execution"""
        try:
            # Create a copy of the frame for thread safety
            frame_copy = gray_frame.copy()
            
            # Calculate target size
            target_width = int(self.frame_width * scale)
            target_height = int(self.frame_height * scale)
            
            # Resize frame (create new buffer to avoid race conditions)
            small_frame = cv2.resize(frame_copy, (target_width, target_height))
            
            faces_and_scores = []
            
            # Detect frontal faces at this scale
            faces = self.face_cascade.detectMultiScale(
                small_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Scale back to original size and add confidence scores
            for (x, y, w, h) in faces:
                scaled_x = int(x / scale)
                scaled_y = int(y / scale)
                scaled_w = int(w / scale)
                scaled_h = int(h / scale)
                
                # Simple confidence based on face size (larger faces = higher confidence)
                confidence = min(1.0, (scaled_w * scaled_h) / (100 * 100))
                faces_and_scores.append(((scaled_x, scaled_y, scaled_w, scaled_h), confidence))
            
            # Try profile detection if no frontal faces at this scale
            if len(faces) == 0:
                profile_faces = self.profile_cascade.detectMultiScale(
                    small_frame,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20)
                )
                
                for (x, y, w, h) in profile_faces:
                    scaled_x = int(x / scale)
                    scaled_y = int(y / scale)
                    scaled_w = int(w / scale)
                    scaled_h = int(h / scale)
                    
                    confidence = min(0.8, (scaled_w * scaled_h) / (100 * 100))  # Slightly lower confidence for profile
                    faces_and_scores.append(((scaled_x, scaled_y, scaled_w, scaled_h), confidence))
            
            return faces_and_scores
            
        except Exception as e:
            print(f"Error in parallel detection at scale {scale}: {e}")
            return []

    def detect_features_parallel(self, frame, face_rects):
        """Detect features for multiple faces in parallel (conservative approach)"""
        if not self.use_multithreading or len(face_rects) <= 2:
            # Fall back to sequential processing for small numbers of faces
            total_eyes = 0
            total_smiles = 0
            for face_rect in face_rects:
                eyes, smiles = self.detect_features(frame, face_rect)
                total_eyes += eyes
                total_smiles += smiles
            return total_eyes, total_smiles
        
        try:
            # Create thread pool only when needed
            if self.thread_pool is None:
                self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
            
            # Submit feature detection tasks for each face using thread-safe method
            futures = []
            for face_rect in face_rects:
                future = self.thread_pool.submit(self.detect_features_safe, frame, face_rect)
                futures.append(future)
            
            # Collect results with timeout
            total_eyes = 0
            total_smiles = 0
            for future in futures:
                try:
                    eyes, smiles = future.result(timeout=0.1)  # 100ms timeout per face
                    total_eyes += eyes
                    total_smiles += smiles
                except Exception as e:
                    # Silently continue on timeout/error
                    continue
            
            return total_eyes, total_smiles
            
        except Exception as e:
            print(f"Parallel feature detection failed, using sequential: {e}")
            # Fall back to sequential
            total_eyes = 0
            total_smiles = 0
            for face_rect in face_rects:
                eyes, smiles = self.detect_features(frame, face_rect)
                total_eyes += eyes
                total_smiles += smiles
            return total_eyes, total_smiles

    def detect_features_safe(self, frame, face_rect):
        """Thread-safe feature detection for a single face"""
        try:
            x, y, w, h = face_rect
            
            # Bounds checking to prevent array index errors
            if (y + h > self.gray_frame.shape[0] or x + w > self.gray_frame.shape[1] or 
                x < 0 or y < 0 or w <= 0 or h <= 0):
                return 0, 0
            
            # Create a copy of the face region for thread safety
            face_roi_gray = self.gray_frame[y:y+h, x:x+w].copy()
            
            # Additional safety check for empty ROI
            if face_roi_gray.size == 0:
                return 0, 0
            
            # Create new cascade classifiers for thread safety
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5, minSize=(10, 10))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
            
            # Detect smile
            smiles = smile_cascade.detectMultiScale(face_roi_gray, 1.8, 20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 255), 2)
                cv2.putText(frame, "SMILE!", (x+sx, y+sy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return len(eyes), len(smiles)
            
        except Exception as e:
            print(f"Error in thread-safe feature detection: {e}")
            return 0, 0

def main():
    detector = FaceDetector()
    
    try:
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, detector.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, detector.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Enhanced Face Detection with Multi-scale NMS and Dynamic Optimization")
        print("Press 'e' to cycle effects, 'q' to quit")
        
        frame_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
                
            start_time = time.time()
            detector.frame_count += 1
            frame_counter += 1
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to grayscale once and reuse (optimization)
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=detector.gray_frame)
            
            # Detect motion using background subtraction
            motion_detected = detector.detect_motion(detector.gray_frame)
            
            # Get motion pixels count for adaptive learning (reuse from detect_motion)
            # We'll calculate this from the motion_detected boolean and threshold
            motion_pixels = detector.motion_threshold + 100 if motion_detected else detector.motion_threshold - 100
            
            # Adaptive background learning
            detector.adaptive_background_learning(motion_pixels)
            
            # Dynamic frame skipping and motion-based detection
            should_detect = (detector.frame_count % detector.current_frame_skip == 0) and motion_detected
            
            if should_detect:
                # Multi-scale face detection with NMS
                faces = detector.detect_faces_multiscale(detector.gray_frame)
                detector.last_faces = faces
                detector.last_detection_time = time.time()
            else:
                # Use cached faces if detection was recent (within 2 seconds)
                if time.time() - detector.last_detection_time > 2.0:
                    detector.last_faces = []
            
            # Apply effects to detected faces
            total_eyes = 0
            total_smiles = 0
            
            # Apply effects to each face
            for i, (x, y, w, h) in enumerate(detector.last_faces):
                # Use different colors for multiple faces
                color = detector.colors[i % len(detector.colors)]
                
                # Apply current effect
                detector.apply_face_effect(frame, (x, y, w, h), color, detector.face_effects[detector.current_effect])
            
            # Detect facial features for all faces in parallel
            if detector.last_faces:
                total_eyes, total_smiles = detector.detect_features_parallel(frame, detector.last_faces)
            
            # Calculate FPS and adjust frame skipping
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            detector.fps_queue.append(fps)
            avg_fps = sum(detector.fps_queue) / len(detector.fps_queue)
            
            # Dynamic frame skip adjustment
            detector.adjust_frame_skip_dynamically(avg_fps)
            
            # Adaptive quality control every 30 frames
            if frame_counter % 30 == 0:
                detector.adaptive_quality_control(avg_fps)
            
            # Add UI elements
            detector.add_ui_elements(frame, avg_fps, len(detector.last_faces), motion_detected)
            
            # Show frame
            cv2.imshow('Enhanced Face Detection - Multi-scale NMS', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                detector.current_effect = (detector.current_effect + 1) % len(detector.face_effects)
                print(f"Effect changed to: {detector.face_effects[detector.current_effect]}")
            elif key == ord('t'):
                # Toggle threading
                detector.use_multithreading = not detector.use_multithreading
                if detector.use_multithreading and detector.thread_pool is None:
                    detector.thread_pool = ThreadPoolExecutor(max_workers=detector.max_workers)
                elif not detector.use_multithreading and detector.thread_pool is not None:
                    detector.thread_pool.shutdown(wait=False)
                    detector.thread_pool = None
                print(f"Multi-threading: {'ENABLED' if detector.use_multithreading else 'DISABLED'}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        
        # Cleanup thread pool
        if hasattr(detector, 'thread_pool') and detector.thread_pool is not None:
            detector.thread_pool.shutdown(wait=True)
            print("Thread pool cleaned up")
        
        print("Cleanup completed")

if __name__ == "__main__":
    main()