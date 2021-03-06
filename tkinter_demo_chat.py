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


frame = Frame(window, width=500, height=500)
frame.place(x=20,y=20)
frame.pack_propagate(False) # prevent frame to resize to the labels size

input_field.bind("<Return>", enter_pressed)
frame.pack()

window.mainloop()
