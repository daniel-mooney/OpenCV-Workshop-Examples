from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import cv2 as cv

def grab_frame(capture) -> cv.Mat:
    """
    Return a flipped RGB image from passed capture
    """
    _, frame = capture.read()
    frame = cv.flip(frame, 1)
    return cv.cvtColor(frame, cv.COLOR_BGR2RGB)

def update(fig_frame, capture, im) -> None:
    frame = grab_frame(capture)
    im.set_data(frame)

def main() -> None:
    # Initialise 
    capture = cv.VideoCapture(1)    # front cam

    ax1 = plt.subplot(111)
    im1 = ax1.imshow(grab_frame(capture))
    
    ani = FuncAnimation(plt.gcf(), update, fargs=(capture, im1), interval=1)
    plt.show()

if __name__ == "__main__":
    main()