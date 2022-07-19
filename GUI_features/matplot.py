import matplotlib.pyplot as plt
import cv2 as cv

def main() -> None:
    img = cv.imread("Images\Me.jpg")        # read in image

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)    # convert colourspace

    # Create plot
    plt.imshow(img_rgb)
    plt.xticks([]), plt.yticks([])      # Remove axe ticks

    plt.title("Myself")
    plt.xlabel("Neck Girth")

    plt.show()

if __name__ == "__main__":
    main()