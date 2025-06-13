import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
from threading import Thread
import time

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Create a directory to store the images if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("400x200")
        self.is_matching = False

        # Capture Face Button
        self.capture_button = tk.Button(root, text="Add User", command=self.capture_face_images)
        self.capture_button.pack(pady=10)

        # Match Face Button
        self.match_button = tk.Button(self.root, text="Verify User", width=20, command=self.match_face)
        self.match_button.pack(pady=10)

        # Exit Button
        self.exit_button = tk.Button(root, text="Exit", width=20, command=self.root.quit)
        self.exit_button.pack(pady=10)
      
    def capture_face_images(self):
        # Use after() to call the dialog in the main thread
        self.root.after(0, self.capture_face_image)

    def capture_face_image(self):
        name = simpledialog.askstring("Input", "Enter your name:")
        if not name:
            messagebox.showerror("Error", "Name is required")
            return
        
        cap = cv2.VideoCapture(0)
        img_count = 0
        
        messagebox.showinfo("Instructions", f"Capturing face images for {name}.\nLook at the camera and press 'q' to quit early.")
        
        while img_count < 20:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture from camera")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw rectangle around the face and save the image
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract face region
                face_image = frame[y:y+h, x:x+w]
                
                # Save the face image
                img_filename = f'dataset/{name}_{img_count + 1}.jpg'
                cv2.imwrite(img_filename, face_image)
                img_count += 1
                
                # Show progress
                cv2.putText(frame, f'Captured: {img_count}/20', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Capturing Face Images', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", f"Face images saved for {name}! Total captured: {img_count}")

    def match_face(self):
        if self.is_matching:
            messagebox.showwarning("Warning", "Face verification is already running!")
            return
            
        # Check if dataset exists and has images
        if not os.path.exists('dataset') or not os.listdir('dataset'):
            messagebox.showerror("Error", "No face images found in dataset. Please add users first.")
            return
            
        def match():
            self.is_matching = True
            cap = cv2.VideoCapture(0)
            
            while self.is_matching:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                start_time = time.time()
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    face_image = gray[y:y+h, x:x+w]
                    best_match = None
                    min_distance = float('inf')
                    
                    # Compare with stored faces
                    for filename in os.listdir('dataset'):
                        if filename.endswith('.jpg'):
                            stored_face = cv2.imread(os.path.join('dataset', filename))
                            if stored_face is not None:
                                stored_face_gray = cv2.cvtColor(stored_face, cv2.COLOR_BGR2GRAY)
                                resized_stored_face = cv2.resize(stored_face_gray, (w, h))
                                
                                # Compute the euclidean distance
                                dist = np.linalg.norm(resized_stored_face - face_image)
                                if dist < min_distance:
                                    min_distance = dist
                                    best_match = filename
                    
                    # Display result
                    if best_match and min_distance < 5000:  # Threshold for matching
                        user_name = best_match.split('_')[0]
                        label = f"{user_name} ({min_distance:.0f})"
                        color = (0, 255, 0)  # Green for match
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)  # Red for no match
                    
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Calculate and display FPS
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display the frame
                cv2.imshow('Face Verification', frame)
                
                # Break on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            self.is_matching = False
        
        # Start matching in a separate thread
        match_thread = Thread(target=match)
        match_thread.daemon = True  # Dies when main thread dies
        match_thread.start()
    
    def stop_matching(self):
        self.is_matching = False
        cv2.destroyAllWindows()

# Create the GUI application window
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()