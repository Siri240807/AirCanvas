import cv2
import mediapipe as mp

mp_h = mp.solutions.hands
mp_d = mp.solutions.drawing_utils

hands = mp_h.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('AirCanvasV.mp4', fcc, 20.0, (frame_w, frame_h))

canvas = None
drawing = False  #starting canvas with False..
prev_x, prev_y = None, None
color = (0, 0, 255)  #starting with red...
thickness = 5
eraser_mode = False

print("Air Canvas")
print("Keys: [W]=Write  [C]=Clear  [E]=Eraser  [1/2/3]=Colors  [ESC]=Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not detected!!!")
        break
    
    frame = cv2.flip(frame, 1)
    
    
    if canvas is None:
        canvas = frame.copy() * 0
    
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    h, w, _ = frame.shape
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_d.draw_landmarks(frame, hand_landmarks, mp_h.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            if drawing:
                if prev_x is not None and prev_y is not None:
                    if eraser_mode:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 0), 50)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), (x, y), color, thickness)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None
    else:
        prev_x, prev_y = None, None

    
    frame = cv2.addWeighted(frame, 0.6, canvas, 0.8, 0)

    
    mode_text = "Eraser" if eraser_mode else "Pen"
    cv2.putText(frame, f"Mode: {mode_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    cv2.putText(frame, f"Writing: {'ON' if drawing else 'OFF'}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 255, 0) if drawing else (0, 0, 255), 2)
    
    cv2.imshow("Air Writing (with Colors & Eraser)", frame)
    out.write(frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  #ESC key
        break
    elif key in [ord('w'), ord('W')]:
        drawing = not drawing
    elif key in [ord('c'), ord('C')]:
        canvas = frame.copy() * 0
    elif key in [ord('e'), ord('E')]:
        eraser_mode = not eraser_mode
    elif key == ord('1'):
        color, eraser_mode = (0, 0, 255), False  #red
    elif key == ord('2'):
        color, eraser_mode = (0, 255, 0), False  #green
    elif key == ord('3'):
        color, eraser_mode = (255, 0, 0), False  #blue

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video saved as 'AirCanvasV.mp4'")

