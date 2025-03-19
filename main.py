import face_recognition
import cv2
import numpy as np
import os


class FaceRecognitionSystem:
    def __init__(self, faces_dir="faces"):
        """Initialize the face recognition system.

        Args:
            faces_dir: Directory containing face images for recognition.
                       Each image should be named as "person_name.jpg".
        """
        self.video_capture = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_locations = []
        self.face_names = []
        self.process_this_frame = True

        # Load known faces from directory
        self.load_known_faces(faces_dir)

    def load_known_faces(self, faces_dir):
        """Load known faces from the specified directory."""
        if not os.path.exists(faces_dir):
            print(f"Directory '{faces_dir}' not found. Creating it...")
            os.makedirs(faces_dir)
            print(f"Please add your face images to '{faces_dir}' directory.")
            return

        print(f"Loading known faces from '{faces_dir}'...")
        for filename in os.listdir(faces_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get person name from filename (without extension)
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(faces_dir, filename)

                try:
                    # Load image and get face encoding
                    image = face_recognition.load_image_file(image_path)
                    face_encodings = face_recognition.face_encodings(image)

                    if len(face_encodings) > 0:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        print(f"Loaded face: {name}")
                    else:
                        print(f"No face found in image: {filename}")
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")

        print(f"Loaded {len(self.known_face_names)} faces")

    def start_video(self, camera_index=0):
        """Start video capture from specified camera."""
        self.video_capture = cv2.VideoCapture(camera_index)
        if not self.video_capture.isOpened():
            print("Error: Could not open video capture device.")
            return False
        return True

    def process_frame(self, frame):
        """Process a video frame for face recognition."""
        # Only process every other frame to save processing power
        if self.process_this_frame:
            # Resize frame to 1/4 size for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert from BGR to RGB (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Find faces in the current frame
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in face_encodings:
                # Compare with known faces
                if len(self.known_face_encodings) > 0:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"

                    # Use the closest match
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] <= 0.45:
                        name = self.known_face_names[best_match_index]

                    self.face_names.append(name)
                else:
                    self.face_names.append("Unknown - No known faces loaded")

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

        return frame

    def run(self):
        """Run the face recognition system."""
        if not self.start_video():
            return

        print("Starting face recognition. Press 'q' to quit.")

        while True:
            # Capture frame-by-frame
            ret, frame = self.video_capture.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Process this frame
            frame = self.process_frame(frame)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close windows
        self.video_capture.release()
        cv2.destroyAllWindows()
        print("Face recognition stopped")


if __name__ == "__main__":
    # Create and run the face recognition system
    face_system = FaceRecognitionSystem()
    face_system.run()
