import cv2

image = cv2.imread("body.jpg")

body_cascade = cv2.CascadeClassifier("fullbody.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

bodies = body_cascade.detectMultiScale(gray, 1.2, 1)

for body in bodies:
    (x, y, w, h) = body
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
