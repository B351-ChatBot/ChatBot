import tkinter as tk

class Chat:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chat Bot")
        
        self.entryText = tk.StringVar()
        self.e = tk.Entry(self.root, textvariable=self.entryText, bd=5)
        self.e.bind('<Return>', self.user_response(self.e.get()))
        self.e.pack(side="left")

        # Links 17 through 26 are comminted out untill I can get the textbox to take the enter key.
        # could not get to take this button below either.
        # Link 17 will be used eventually, for now i have been working on getting it to print the output in the shell I'm useing.
        
        #self.w = tk.Message(self.root, ...)
        
        '''     
        self.hi_there = tk.Button(self.root)
        self.hi_there["text"]="Reply"
        self.hi_there["command"]=self.user_response()
        self.hi_there.pack(side="right")

        print("working")
        '''
        self.root.mainloop()

    def data_access(self):
        print("")

    def user_response(self, event):
        print("working")
        print(event.widget.get())

    def ai_response(self):
        print("")
    

Chat()
