import tkinter as tk

class Chat:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Chat Bot")

        self.w = tk.Message(self.root, text="", width=50)

        self.e = tk.Entry(self.root, bd=5, width=50)
        self.e.bind(sequence='<Return>', func=self.user_response)
        self.e.pack(side="left")
        
        self.root.mainloop()

    def data_access(self):
         x = sample(l_words, 1)
         self.w.configure(text=self.x)

    def user_response(self, event):
        self.w.configure(text=self.e.get())

    def ai_response(self):
        print("")
    

Chat()
