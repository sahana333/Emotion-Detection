import cv2
from deepface import DeepFace
import numpy as np

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        try:
            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Analyze emotions using DeepFace
            analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)

            # Check if the result is a list (handle accordingly)
            if isinstance(analysis, list):
                analysis = analysis[0]

            # Get the dominant emotion
            dominant_emotion = analysis.get('dominant_emotion', 'No Face Detected')

            # Display the emotion on the video feed
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception as e:
            # Handle errors and display them
            print("Error during analysis:", e)

        # Display the video feed
        cv2.imshow('Emotion Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
