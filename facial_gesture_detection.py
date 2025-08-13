import cv2
import dlib
import numpy as np
from scipy.spatial import distance

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  
# Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

def eye_aspect_ratio(eye_points):
    # Compute distances between vertical eye landmarks
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    # Compute distance between horizontal eye landmarks
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_points):
    # Similar to EAR but for mouth
    A = distance.euclidean(mouth_points[13], mouth_points[19])  # vertical
    B = distance.euclidean(mouth_points[14], mouth_points[18])  # vertical
    C = distance.euclidean(mouth_points[12], mouth_points[16])  # horizontal
    mar = (A + B) / (2.0 * C)
    return mar

def eyebrow_raise_ratio(eyebrow_points, eye_points):
    # Measure vertical distance between eyebrow and eye
    # Average vertical difference between eyebrow pts and eye pts
    distances = []
    for eb, e in zip(eyebrow_points, eye_points):
        distances.append(abs(eb[1] - e[1]))
    return np.mean(distances)

def is_red_pixel(bgr_pixel, threshold=50):
    b, g, r = bgr_pixel
    return (r - max(g, b)) > threshold and r > 100

def detect_pimples(frame, face_rect):
    # Simple heuristic: detect small red spots in face region
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    face_roi = frame[y:y+h, x:x+w]
    pimples = 0
    # Resize for speed
    small_face = cv2.resize(face_roi, (100, 100))
    for i in range(100):
        for j in range(100):
            if is_red_pixel(small_face[i,j]):
                pimples += 1
    # Threshold number of red pixels to say pimples detected
    return pimples > 50

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# Indices for facial landmarks:
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
MOUTH_POINTS = list(range(48, 68))
LEFT_EYEBROW_POINTS = list(range(22, 27))
RIGHT_EYEBROW_POINTS = list(range(17, 22))

# Thresholds for eye blink and mouth open
EYE_AR_THRESH = 0.2
MOUTH_AR_THRESH = 0.6
EYEBROW_RAISE_THRESH = 15  # heuristic, you can adjust

cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = shape_to_np(shape)

        # Eyes
        left_eye = shape_np[LEFT_EYE_POINTS]
        right_eye = shape_np[RIGHT_EYE_POINTS]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Mouth
        mouth = shape_np[MOUTH_POINTS]
        mar = mouth_aspect_ratio(mouth)

        # Eyebrows
        left_eyebrow = shape_np[LEFT_EYEBROW_POINTS]
        right_eyebrow = shape_np[RIGHT_EYEBROW_POINTS]

        left_eye_pts = shape_np[LEFT_EYE_POINTS]
        right_eye_pts = shape_np[RIGHT_EYE_POINTS]

        left_eyebrow_raise = eyebrow_raise_ratio(left_eyebrow, left_eye_pts)
        right_eyebrow_raise = eyebrow_raise_ratio(right_eyebrow, right_eye_pts)
        eyebrow_raise = (left_eyebrow_raise + right_eyebrow_raise) / 2.0

        # Pimple detection (simple)
        pimples_detected = detect_pimples(frame, face)

        # Draw landmarks
        for (x, y) in shape_np:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Display info
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Eyebrow Raise: {eyebrow_raise:.1f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Gesture detection logic
        if ear < EYE_AR_THRESH:
            cv2.putText(frame, "Blinking", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Mouth Open", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if eyebrow_raise > EYEBROW_RAISE_THRESH:
            cv2.putText(frame, "Eyebrow Raised", (300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if pimples_detected:
            cv2.putText(frame, "Pimples Detected", (300, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Facial Gesture Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
