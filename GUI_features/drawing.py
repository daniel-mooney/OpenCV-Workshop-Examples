import cv2 as cv
import numpy as np

def main() -> None:
    # Create a blank image
    blank = np.zeros((500, 500, 3))

    # Draw Rectangle
    pt1, pt2 = (25, 60), (250, 250)             # Top Left, bottom right
    rect_colour = (0, 0, 255)                   # BGR

    cv.rectangle(blank, pt1, pt2, rect_colour, thickness=2)

    # Draw Circle
    centre = blank.shape[1] // 2, blank.shape[0] // 2   # Centre of img
    radius = 40                                         # Units in pixels
    circle_colour = (0, 125, 80)                        # BGR

    cv.circle(blank, centre, radius, circle_colour, thickness=2)

    # Draw Line
    pt1, pt2 = (80, 450), (400, 200)
    line_colour = (255, 0, 0)

    cv.line(blank, pt1, pt2, line_colour, thickness=4)

    # Write Text
    text = "Hello!"
    org = (250, 450)                    # Bottom left corner of text
    text_colour = (0, 255, 0)           # BGR

    cv.putText(blank, text, org, cv.FONT_HERSHEY_PLAIN, 5, text_colour, thickness=2)

    # Show drawing
    cv.imshow("Drawings", blank)
    cv.waitKey(0)


if __name__ == "__main__":
    main()