from tkinter import *
import ChatBot as c

myCB = c.ChatBot('corpus/movie_lines.txt','corpus/movie_conversations.txt')
window = Tk()

input_user = StringVar()
input_field = Entry(window, text=input_user)
input_field.pack(side=BOTTOM, fill=X)

def enter_pressed(event):
    input_get = input_field.get()
    output_display = input_get + "\n" + myCB.converse(input_get)
    print(input_get)
    print (output_display)
    label = Label(frame, text=output_display)
    input_user.set('')
    label.pack()
    return "break"

def myfunction(event):
    canvas.configure(scrollregion=canvas.bbox("all"),width=200,height=200)

myframe = Frame(window, width=500, height=500)
myframe.place(x=20,y=20)
myframe.pack_propagate(False) # prevent frame to resize to the labels size

canvas=Canvas(myframe)
frame=Frame(canvas)
myscrollbar=Scrollbar(myframe,orient="vertical",command=canvas.yview)
canvas.configure(yscrollcommand=myscrollbar.set)

myscrollbar.pack(side="right",fill="y")
canvas.pack(side="left")
canvas.create_window((0,0),window=frame,anchor='nw')
frame.bind("<Configure>",myfunction)

input_field.bind("<Return>", enter_pressed)
frame.pack()

window.mainloop()
