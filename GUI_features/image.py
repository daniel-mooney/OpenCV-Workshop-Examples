import cv2 as cv

def rescale(frame: cv.Mat, scale: float) -> cv.Mat:
    """
    Returns a rescaled frame

    Parameters:
        frame (cv.Mat): The frame to be rescaled
        scale (float): A float representing a percentage to scale the frame by, i.e 0.7 = 70%.
    
    Returns:
        A Mat object of the new rescaled frame.
    """

    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dim = (width, height)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

def main():
    beach = cv.imread("../Images/Beach.jpg")
    smaller_beach = rescale(beach, 0.75)

    cv.imshow("Beach", beach)                       # Show window
    cv.imshow("Smaller Beach", smaller_beach)
    cv.waitKey(0)                                    # Wait and close window only when key is pressed

if __name__ == "__main__":
    main()