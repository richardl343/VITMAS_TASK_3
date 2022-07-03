import numpy as np
import cv2 as cv
import math
import pyautogui
                                                                             

#to open Camera

capture = cv.VideoCapture(0)

while capture.isOpened():

    # Capture frames from the camera
    ret, frame = capture.read()

    # Get hand data from the rectangle sub window
    cv.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = frame[100:300, 100:300]

    # Apply Gaussian blur
    blur = cv.GaussianBlur(crop_image, (3, 3), 0)

    # Change color-space from BGR -> HSV
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

    # Create a binary image with where white will be skin colors and rest is black
    mask2 = cv.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel for morphological transformation
    kernel = np.ones((5, 5))

    # Apply morphological transformations to filter out the background noise
    dilation = cv.dilate(mask2, kernel, iterations=1)
    erosion = cv.erode(dilation, kernel, iterations=1)

    # Apply Gaussian Blur and Threshold
    filtered = cv.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv.threshold(filtered, 127, 255, 0)
#####
    # Find contours
    #image, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    try:
        # Find contour with maximum area
        contour = max(contours, key=lambda x: cv.contourArea(x))

        # Create bounding rectangle around the contour
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)

        # Find convex hull
        hull = cv.convexHull(contour)

        # Draw contour
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Fi convexity defects
        hull = cv.convexHull(contour, returnPoints=False)
        defects = cv.convexityDefects(contour, hull)

        # Use cosine rule to find angle of the far point from the start and end point i.e. the convex points (the finger
        # tips) for all defects
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # if angle >= 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv.line(crop_image, start, end, [0, 255, 0], 2)

        # Press SPACE if condition is match

        if count_defects >= 4:
                pyautogui.press('space')
                cv.putText(frame, "JUMP", (115, 80), cv.FONT_HERSHEY_SIMPLEX, 2, 2, 2)

        #PLAY RACING GAMES (WASD)
        """
        if count_defects == 1:
            pyautogui.press('w')
            cv.putText(frame, "W", (115, 80), cv.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 2:
            pyautogui.press('s')
            cv.putText(frame, "S", (115, 80), cv.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 3:
            pyautogui.press('aw')
            cv.putText(frame, "aw", (115, 80), cv.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 4:
            pyautogui.press('dw')
            cv.putText(frame, "dw", (115, 80), cv.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        if count_defects == 5:
            pyautogui.press('s')
            cv.putText(frame, "s", (115, 80), cv.FONT_HERSHEY_SIMPLEX, 2, 2, 2)
        """

    except:
        pass

    # Show required images
    cv.imshow("Gesture", frame)

    # Close the camera if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()