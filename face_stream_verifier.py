import cv2
import os
import numpy as np
import time

# Load the face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create necessary directories
if not os.path.exists('dataset'):
    os.makedirs('dataset')

def capture_face_images():
    name = input("Enter your name: ")
    cap = cv2.VideoCapture(0)
    img_count = 0
    
    print(f"Capturing face images for {name}. Press 'q' to quit early.")
    
    while img_count < 20:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process detected faces
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Extract face region
            face_image = frame[y:y+h, x:x+w]
            
            # Save the face image
            img_filename = f'dataset/{name}_{img_count + 1}.jpg'
            cv2.imwrite(img_filename, face_image)
            img_count += 1
            
            # Show count on frame
            cv2.putText(frame, f'Captured: {img_count}/20', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Capturing Face Images', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Face images saved! Total captured: {img_count}")

def match_face():
    # Check if dataset directory exists and has images
    if not os.path.exists('dataset') or not os.listdir('dataset'):
        print("No face images found in dataset. Please capture some faces first.")
        return
        
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
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
                        
                        # Resize stored face to match current face size
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
        cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Verification', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\n=== Face Verification System ===")
        print("1. Capture Face Images")
        print("2. Match Face")
        print("q. Quit")
        choice = input("Enter your choice: ").lower()
        
        if choice == "1":
            capture_face_images()
        elif choice == "2":
            match_face()
        elif choice == "q":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")