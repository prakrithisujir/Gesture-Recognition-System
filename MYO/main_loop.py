#%%
import pandas as pd
from tkinter import *
from PIL import Image, ImageTk
#import pyttsx3
from adafruit_servokit import ServoKit
from tensorflow.keras.models import load_model as tfk__load_model
window = Tk()
window.title("Gesture Classifier")
window.geometry('1024x768')
train_lbl = Label(window, text="Training Set")
train_lbl.grid(column=0, row=0)
train_txt = Entry(window, width=10)
train_txt.grid(column=1, row=0)
test_lbl = Label(window, text="Epochs")
test_lbl.grid(column=0, row=1)
test_txt = Entry(window, width=10)
test_txt.grid(column=1, row=1)
train_op_lbl = Label(window, text="Actual")
train_op_lbl.grid(column=0, row=2)
test_op_lbl = Label(window, text="Prediction")
test_op_lbl.grid(column=0, row=3)

actual_op = Label(window, text=" ")
actual_op.grid(column=1, row=2)
predicted_op = Label(window, text=" ")
predicted_op.grid(column=1, row=3)

serial_lbl = Label(window, text="Serial")
serial_lbl.grid(column=0, row=4)
serial_txt = Entry(window, width=10)
serial_txt.grid(column=0 ,row=5)

word_lbl = Label(window, text="Word:")
word_lbl.grid(column=0, row=7)

def numbers_to_strings(argument):
    switcher = {
        0: "Point",
        1: "Middle",
        2: "Grip",
        3: "Pinch",
        4: "No Gesture"
    }
 
    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(argument, "nothing")

def numbers_to_controls(argument):
    switcher = {
        0: [0,180,0,180,180],
        1: [0,0,180,180,180],
        2: [0,0,0,180,180],
        3: [0,180,0,180,0]
        #3: [90,90,90,0,0]
    }
 
    # get() method of dictionary data type returns
    # value of passed argument if it is present
    # in dictionary otherwise second argument will
    # be assigned as default value of passed argument
    return switcher.get(argument, [180,180,180,0,0])  


def train_click():
    import dl
    file = train_txt.get()
    num_epochs = test_txt.get()
	
    model = dl.train(file,num_epochs)
	
    model.save('model.h5')
	#train_op_lbl.configure(text=model)
    load = Image.open('Model-Accuracy.png')
    render = ImageTk.PhotoImage(load)
    img = Label(image=render, height="500", width="900")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=3, row=100)#, padx=10, pady = 10)


def gesture_click():
    import gesture
    sample_number = serial_txt.get()
    actual,prediction=gesture.generate(int(sample_number))
    actual_op = Label(window, text=actual)
    actual_op.grid(column=1, row=2)
    predicted_op= Label(window, text=prediction)
    predicted_op.grid(column=1, row=3)
    load = Image.open('gesture.png')
    render = ImageTk.PhotoImage(load)
    img = Label(image=render, height="500", width="900")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=3, row=100)#, padx=10, pady = 10)


    
    word_txt = Label(window, text=numbers_to_strings(int(prediction)))
    word_txt.grid(column=2, row=7)
    
    angle=numbers_to_controls(int(prediction))
    print(angle)
    kit=ServoKit(channels=16)
    
    kit.servo[1].angle=angle[1]
    kit.servo[2].angle=angle[2]
    kit.servo[3].angle=angle[3]
    kit.servo[4].angle=angle[4]
    kit.servo[0].angle=angle[0]
    
    #engine = pyttsx3.init()
    #if prediction !=4:
    #    engine.say(numbers_to_strings(int(prediction)))
    #engine.runAndWait()




train_btn = Button(window, text="Train", command=train_click)
train_btn.grid(column=2, row=0)
#test_btn = Button(window, text="Test", command=test_click)
#test_btn.grid(column=2, row=1)

gesture_btn=Button(window, text="Generate Gesture", command=gesture_click)
gesture_btn.grid(column=0, row=6)

#window.attributes('-fullscreen', True)
window.mainloop()

#%%

#%%
