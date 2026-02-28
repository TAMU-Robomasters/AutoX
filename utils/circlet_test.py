import cv2 as cv

def main():
    cap = cv.VideoCapture(1)          # open camera at index 1
    if not cap.isOpened():
        print("failed to open camera 1")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("lost frame")
            break

        cv.imshow("cam1", frame)
        # press q to quit
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
