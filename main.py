import numpy as np
import cv2
import pickle
import tkinter as tk
from tkinter import font
from PIL import ImageTk, Image
###################
width = 640
height = 480
threshold = 0.65
dynamic_text = " "
#####################
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("model_trained_10.p", "rb")
model = pickle.load(pickle_in)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def append_text(pre):
    global dynamic_text
    dynamic_text += str(pre)
    T.config(state=tk.NORMAL)  # Enable editing
    T.insert(tk.END, dynamic_text)
    T.config(state=tk.DISABLED)  # Disable editing to maintain scroll position
    T.see(tk.END)  # Scroll to the end to show the newly appended text

# def sound():
#     playsound('beep.wav')


root = tk.Tk()
root.title("Tkinter OpenCV Demo")
root.geometry("1080x720")
root.config(background='#090970')


#####TEXT-FIELD############################################
T = tk.Text(root, wrap=tk.WORD, height=50, width=40, background='#090970', foreground='white')
scrollbar = tk.Scrollbar(root, command=T.yview)
T.config(yscrollcommand=scrollbar.set)
T.insert(tk.END, dynamic_text)
T.place(x=1200, y=10)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
###################################################

# Create two canvas widgets for displaying the original and processed images
canvas_original = tk.Canvas(root, width=width, height=height, background='black')
canvas_original.place(x=0, y=0)


# canvas_processed = tk.Canvas(root, width=width, height=height)
# canvas_processed.pack(side=tk.LEFT)





def process_frame():
    ret, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    img = img.reshape(1, 32, 32, 1)

    # Predict
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(predictions)
    append_text(predictions)

    # Display processed image with prediction
    if probVal > threshold:
        lbl = tk.Label(root, text="Digit" + ": " + str(classIndex) + " prob_Value: " + str(probVal) , foreground='white', background='#090970',font= font.Font(size=24))
        lbl.place(x=700, y=100)
        cv2.putText(imgOriginal, str(classIndex) + "  " + str(probVal), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    # Convert the processed image from OpenCV BGR to RGB format
    img_rgb = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Update canvas with new image
    canvas_original.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas_original.img_tk = img_tk  # Save a reference to avoid garbage collection issues

    # Convert the processed image to grayscale for display
    img_processed = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)
    img_processed_pil = Image.fromarray(img_processed)
    img_processed_tk = ImageTk.PhotoImage(image=img_processed_pil)

    # Update canvas with new processed image
    # canvas_processed.create_image(0, 0, anchor=tk.NW, image=img_processed_tk)
    # canvas_processed.img_processed_tk = img_processed_tk  # Save a reference to avoid garbage collection issues

    # Schedule the next frame update
    root.after(10, process_frame)


# Start processing frames
process_frame()

root.mainloop()

# Release the camera and close all OpenCV windows when the Tkinter window is closed
cap.release()
cv2.destroyAllWindows()
