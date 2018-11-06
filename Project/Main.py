import tkinter as tk
from tkinter import font as tkfont
import cv2
from PIL import Image
from PIL import ImageTk
from tkinter import filedialog
from imutils import contours
import numpy as np
import imutils
from tkinter import messagebox


def client_exit():  # exit function for closing gui
    exit()


def combine_funcs(*funcs):
    def combined_func(*args, **kwargs):
        for f in funcs:
            f(*args, **kwargs)
    return combined_func


def face_and_eye():  # function for face and eye detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # importing the haar cascade suitable for face

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # importing the haar cascade suitable for eye

    cap = cv2.VideoCapture(0)   # We enter 0 for capturing through laptop camera
    # and 1 if using external camera

    while 1:  # will run till condition is satisfied
        ret, img = cap.read()   # read camera
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # converting image to grayscale
        # we convert to grayscale because of pixel density
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # detecting objects in form of rectangles

        font = cv2.FONT_HERSHEY_SIMPLEX  # declaring font of out choice

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)   # creating rectangle in image(img), coordinates
            # for vertex p1 (x,y), co-ordinates of P2 vertex opposite to P1 (x + w, y + h), BGR value (blue), thickness
            roi_gray = gray[y:y + h, x:x + w]  # it crops the part of image based on co-ordinates from detectMultiScale
            roi_color = img[y:y + h, x:x + w]  # same as above
            cv2.putText(img, "Face", (0, 50), font, 2, (0, 0, 255), 1, cv2.LINE_AA)  # displaying text when detected

            eyes = eye_cascade.detectMultiScale(roi_gray)  # gets co-ordinates of eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2, lineType=5)
                # the above line is same as that we created for face detection
                cv2.putText(roi_color, "Eye",(0, 50), font, 2, (0, 0, 255), 1, cv2.LINE_AA)
                # cv2.circle(roi_color, (ex , ey + eh), int(((eh)**2 + (ew)**2)**(0.5)), (0,255,0), 2, lineType=8)

        cv2.imshow('img', img)   # this displays or shows the img window
        k = cv2.waitKey(30) & 0xff  # The function waitKey() waits for key event for a "delay" (here, 30 milliseconds)
        if k == 27:  # Exc key to stop
            break

    cap.release()  # release the software or hardware resource(here, camera)
    # if you don't release the camera you can't close your camera window, it gets struck for a while when esc is pressed
    cv2.destroyAllWindows()  # destroys all windows


def select_image():  # function for edge covering

    global panelA, panelB  # globalised panels are created

    path = filedialog.askopenfilename()  # path for selecting the file

    if len(path) > 0:  # this takes care for invalid path

        image = cv2.imread(path)  # imread attribute reads the image, you should know that cv2 reads in BGR
        #  format only (Blue-Green-Red)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # this converts BGR to grayscale
        edged = cv2.Canny(gray, 50, 100)   # Canny edge detector is an edge detection operator that uses a multi-stage
        # algorithm to detect a wide range of edges in images

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # we convert BGR to RGB as PIL reads image in RGB format only

        image = Image.fromarray(image)   # converts into binary data
        image = image.resize((500, 500), Image.ANTIALIAS)  # resizing the image in case it is a large one
        edged = Image.fromarray(edged)   # converting edged image data to binary data
        edged = edged.resize((500, 500), Image.ANTIALIAS)   # resizing

        image = ImageTk.PhotoImage(image)   # Tkinter-compatible photo image in PIL library
        edged = ImageTk.PhotoImage(edged)   # same to the edged window

        if panelA is None or panelB is None:   # if nothing shown in both tkinter windows

            panelA = tk.Label(image=image)   # Displaying original image in PanelA
            # A label is a widget class in Tkinter
            panelA.Image = image
            panelA.pack(side="left", padx=10, pady=10)   # packing it

            panelB = tk.Label(image=edged)   # displaying edged image in PanelB
            panelB.Image = edged
            panelB.pack(side="right", padx=10, pady=10)    # packing it

        else:
            panelA.configure(image=image)   # If nothing detected
            panelB.configure(image=edged)
            panelA.Image = image
            panelB.Image = edged


panelA = None
panelB = None


def motion_sensor():  # this function is for motion sensor

    def diffimg(a, b, c):  # we created this to differentiate between 3 instants of image
        t0 = cv2.absdiff(a, b)  # absdiff = absolute difference
        t1 = cv2.absdiff(b, c)  # instants taken at time t,t+1,t+2

        t3 = cv2.bitwise_and(t0, t1)
        return t3  # returns a boolean

    cap = cv2.VideoCapture(0)   # capturing through camera
    t = cap.read()[1]  # reads camera and stores in t
    tp = cap.read()[1]  # stores in t-plus(tp)
    tpp = cap.read()[1]  # stores in t-plus-plus(tpp)

    t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)  # converts to grayscale because of good pixel density
    tp = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
    tpp = cv2.cvtColor(tpp, cv2.COLOR_BGR2GRAY)

    while True:
        img = diffimg(t, tp, tpp)  # check image instants for checking motion

        cv2.imshow('motion sensor', img)  # showing window
        res, img = cap.read()  # reads through camera, res is a boolean
        t = tp  # updating instants
        tp = tpp
        tpp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to grayscale

        key = cv2.waitKey(10)  # waitKey for 10 milliseconds

        if key == 27:   # waiting for esc key to be pressed
            break

    cap.release()
    cv2.destroyAllWindows()  # destroying all windows


# for the following code I am skipping the face and eye detection part as I have discussed it before
def smile_detection():   # this function is for smile detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
    # above line imports haar-cascade file for smile detection

    cap = cv2.VideoCapture(0)
    cap.set(3, 650)  # it is a property of videoCapture whose parameter are property Identifier and value

    while 1:
        ret, img = cap.read()  # read the camera
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converts to grayscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(55, 55),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  #
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            smile = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25),
                                                   flags=cv2.CASCADE_SCALE_IMAGE)

            for (sx, sy, sw, sh) in smile:
                print("Found", len(smile), "smiles!")  # calculates no. of smiles and prints it
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2, lineType=5)  # rectangle creation

            eyes = eye_cascade.detectMultiScale(roi_gray)  # co-ordinates of the grayed eye instance
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2, lineType=5)
                # cv2.circle(roi_color, (ex , ey + eh), int(((eh)**2 + (ew)**2)**(0.5)), (0,255,0), 2, lineType=8)

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()  # releases the window
    cv2.destroyAllWindows()


def body_detection():  # this function is for full body detection

    # def inside(r, q):
    #     rx, ry, rw, rh = r
    #     qx, qy, qw, qh = q
    #     return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def draw_detections(img, rects, thickness=1):   # draws where detected
        for (x, y, w, h) in rects:
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)  # initialising co-ordinates of P1 & P2 of rectangle
            cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)
            # drawing rectangle

    if __name__ == '__main__':  # if the python interpreter is running that module (the source file) as the main
        # program, it sets the special __name__ variable to have a value "__main__". If this file is being imported
        # from another module, __name__ will be set to the module's name

        hog = cv2.HOGDescriptor()  # HOG is Histogram of Oriented Gradient
        # to read more about it go to https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # cap = cv2.VideoCapture(0)
        path = filedialog.askopenfilename()
        cap = cv2.VideoCapture(path)
        # cap = cv2.VideoCapture('C:/Users/lenovo/Desktop/Walking.mp4')

        if cap.isOpened() is True:
            while True:
                _, frame = cap.read()
                found, w = hog.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05)
                draw_detections(frame, found)
                cv2.imshow('feed', frame)
                ch = 0xFF & cv2.waitKey(1)
                if ch == 27:
                    break
            cv2.destroyAllWindows()
        else:
            print("Invalid File Chosen")


def upper_body_detection():

    body_cascade = cv2.CascadeClassifier('haarcascade_mcs_upperbody.xml')

    video_capture = cv2.VideoCapture(0)

    while True:

        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        body = body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in body:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Video', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


def ninja_eye_detector():

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    minf = 100
    maxf = 300

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame_big = cap.read()

        if ret is True:

            scale = 640.0 / frame_big.shape[1]

            frame1 = cv2.resize(frame_big, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            # Resizing the image

            frame_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, flags=0,
                                                  minSize=(minf, minf),
                                                  maxSize=(maxf, maxf))
            for i in range(0, len(faces)):
                x, y, w, h = faces[i]
                ex, ey, ew, eh = int(x + 0.125 * w), int(y + 0.25 * h), int(0.75 * w), int(0.25 * h)
                cv2.rectangle(frame1, (ex, ey), (ex + ew, ey + eh), (128, 255, 0), 2)

            cv2.imshow('frame', frame1)
            key = cv2.waitKey(30)

            if key == 27:
                break
        else:
            break
    cap.release()

    cv2.destroyAllWindows()


def card_reader():
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="path to input image")
    # ap.add_argument("-r", "--reference", required=True,	help="path to reference OCR-A image")
    # args = vars(ap.parse_args())

    # define a dictionary that maps the first digit of a credit card
    # number to the credit card type
    FIRST_NUMBER = {
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
    }

    """ load the reference OCR-A image from disk, convert it to grayscale, and threshold it, such that the digits appear 
    as *white* on a *black* background and invert it, such that the digits appear as *white* on a *black*  """
    # ref = cv2.imread(args["reference"])

    ref = cv2.imread("reference.png")
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

    # find contours in the OCR-A image (i.e,. the outlines of the digits)
    # sort them from left to right, and initialize a dictionary to map
    # digit name to the ROI
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
    refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
    digits = {}

    # loop over the OCR-A reference contours
    for (i, c) in enumerate(refCnts):
        # compute the bounding box for the digit, extract it, and resize
        # it to a fixed size
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))

        # update the digits dictionary, mapping the digit name to the ROI
        digits[i] = roi

    # initialize a rectangular (wider than it is tall) and square
    # structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # load the input image, resize it, and convert it to grayscale
    path = filedialog.askopenfilename()
    image = cv2.imread(path)
    image = imutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply a tophat (whitehat) morphological operator to find light
    # regions against a dark background (i.e., the credit card numbers)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # compute the Scharr gradient of the tophat image, then scale
    # the rest back into the range [0, 255]
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    # apply a closing operation using the rectangular kernel to help
    # cloes gaps in between credit card number digits, then apply
    # Otsu's thresholding method to binarize the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # apply a second closing operation to the binary image, again
    # to help close gaps between credit card number regions
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

    # find contours in the thresholded image, then initialize the
    # list of digit locations
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    locs = []

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if (ar > 2.5) and (ar < 4.0):
            if ((w > 40) and (w < 55)) and ((h > 10) and (h < 20)):
                locs.append((x, y, w, h))

    # sort the digit locations from left-to-right, then initialize the
    # list of classified digits
    locs = sorted(locs, key=lambda x: x[0])
    output = []

    # loop over the 4 groupings of 4 digits
    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []

        # extract the group ROI of 4 digits from the grayscale image,
        # then apply thresholding to segment the digits from the
        # background of the credit card
        group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
        group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # detect the contours of each individual digit in the group,
        # then sort the digit contours from left to right
        digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

        # loop over the digit contours
        for c in digitCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = group[y:y + h, x:x + w]
            roi = cv2.resize(roi, (57, 88))
            # initialize a list of template matching scores
            scores = []

            # loop over the reference digit name and digit ROI
            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                (_, score, _, _) = cv2.minMaxLoc(result)
                scores.append(score)
            groupOutput.append(str(np.argmax(scores)))

        cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # update the output digits list

        output.extend(groupOutput)

    # display the output credit card information to the screen
    # print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    # print("Credit Card #: {}".format("".join(output)))
    output_final = "Card Type : %s \n" % (FIRST_NUMBER[output[0]])
    output_final += "Credit Card Number : %s" % ("".join(output))

    cv2.imshow("Image", image)

    messagebox.showinfo("Output", output_final)

    cv2.waitKey(0)


class OpenCVApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        tk.Tk.wm_title(self, "Open CV Planet")

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold")

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, FaceAndEye, BodySensor, CardReader):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):

        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Welcome to OpenCV Planet!!!", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="Face and Eye Detection",
                            command=lambda: controller.show_frame("FaceAndEye"))
        button2 = tk.Button(self, text="Motion Sensor",
                            command=motion_sensor)
        button3 = tk.Button(self, text="Smile Detector",
                            command=smile_detection)
        button4 = tk.Button(self, text="Body Detection",
                            command=lambda: controller.show_frame("BodySensor"))
        button5 = tk.Button(self, text="Edge Coloring - Photo Editor",
                            command=select_image)
        button6 = tk.Button(self, text="Text Recogniser",
                            command=lambda: controller.show_frame("CardReader"))

        exit_btn = tk.Button(self, text="Exit", fg="red", command=client_exit)

        button1.pack()
        button2.pack()
        button3.pack()
        button4.pack()
        button5.pack()
        button6.pack()
        exit_btn.pack()


class FaceAndEye(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Face and Eye Detection", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button0 = tk.Button(self, text="Normal Mode", command=face_and_eye)
        button0.pack()
        button1 = tk.Button(self, text="Ninja Mode", command=ninja_eye_detector)
        button1.pack()

        button = tk.Button(self, text="Back", fg="red",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


class BodySensor(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button1 = tk.Button(self, text="Upper Body",
                            command=upper_body_detection)
        button1.pack()
        button2 = tk.Button(self, text="Full Body Detection",
                            command=body_detection)
        button2.pack()
        button = tk.Button(self, text="Back", fg="red",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


class CardReader(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="Face and Eye Detection", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button0 = tk.Button(self, text="Card Reader", command=card_reader)
        button0.pack()

        button = tk.Button(self, text="Back", fg="red",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


if __name__ == "__main__":
    app = OpenCVApp()
    app.mainloop()
