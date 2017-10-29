import tkinter as tk

class Chat:
    def __init__(self):
        self.root = tk.Tk() # helps with framework GUI setup
        self.root.title("Chat Bot")  # title
        
        self.entryText = tk.StringVar()  
        self.e = tk.Entry(self.root, textvariable=self.entryText, bd=5) # textbox (can't seem to get this to pass the valus, the issue is with this)
        self.e.bind('<Return>', self.user_response(self.e.get())) # for keyboard (allows users to press enter to send message)
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

    # for Database accessing (wasn't sure if we would need this)
    def data_access(self):
        print("")

    # for users reponse
    # will print the user's message on screen eventually, just prints to shell/commandline for now
    def user_response(self, event):
        print("working")
        print(event.widget.get())

    # for AI response
    # should randomly pick a work and otuput it. Will use a list untill database is set up.
    def ai_response(self):
        print("")
    

Chat()
