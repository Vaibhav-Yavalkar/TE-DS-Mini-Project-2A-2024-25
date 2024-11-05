import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_distance(a, b):
    """Calculate the Euclidean distance between two points."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

def jumping_jacks_frames():
    """Detect and count jumping jacks using webcam feed."""
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    # Initialize the Pose solution
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            # Convert the RGB image back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract pose landmarks
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, 
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                # Calculate the distance between the shoulders
                shoulder_distance = calculate_distance(left_shoulder, right_shoulder)

                # Update stage and count jumps
                if shoulder_distance > 0.200:  # Threshold for arms raised
                    stage = "up"
                if shoulder_distance < 0.170 and stage == 'up':  # Threshold for arms lowered
                    stage = "down"
                    counter += 1

                # Draw pose landmarks and connections on the image
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Display the shoulder distance and rep counter on the image
                cv2.putText(image, f'Shoulder Distance: {int(shoulder_distance * 1000)}', 
                            (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
            except Exception as e:
                print(e)

            # Display the rep counter on the image
            cv2.putText(image, 'REPS', (50,50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (50,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)

            # Encode the image as JPEG and yield it for streaming
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
