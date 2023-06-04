import cv2

cap = cv2.VideoCapture("body.mp4")

body_cascade = cv2.CascadeClassifier("fullbody.xml")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_cascade.detectMultiScale(gray, 1.2, 4)

    for body in bodies:
        (x, y, w, h) = body
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
