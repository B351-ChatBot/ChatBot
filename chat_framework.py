import tkinter as tk
import random

class Chat:
    def __init__(self):
        self.l = ["hey", "lie", "cat", "dog", "bird", "bat"]
        
        self.root = tk.Tk()
        self.root.title("Chat Bot")

        self.t = ""

        self.e = tk.Entry(self.root, bd=5, width=50)
        self.e.bind(sequence='<Return>', func=self.user_response)
        self.e.pack(side="left")
        
        self.root.mainloop()

    def data_access(self):
         self.t = sample(l_words, 1)
         self.w = tk.Label(self.root, text=self.t, width=50)
         self.w.pack(fill="both", expand=True, padx=10, pady=10)

    def user_response(self, event):
        self.t = self.e.get()
        self.w = tk.Label(self.root, text=self.t, width=50)
        self.w.pack(fill="both", expand=True, padx=10, pady=10)

        self.ai_response()
        

    def ai_response(self):
        num = random.randint(0,5)
        self.t = self.l[num]
        self.w = tk.Label(self.root, text=self.t, width=50)
        self.w.pack(fill="both", expand=True, padx=10, pady=10)
    

Chat()
