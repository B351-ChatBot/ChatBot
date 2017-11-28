# -*- coding: utf-8 -*-
"""
B351 - Project Chatbot

Project Team:
Tiana Deckard
Hans Thieme
David Bickel

"""

import tkinter as tk
import random
from ChatBot import *

class Chat:
    #initialize the list of words and start up the gui
    def __init__(self):

        self.myCB = ChatBot("corpus/movie_lines.txt","corpus/movie_conversations.txt")
        
        self.root = tk.Tk()
        self.root.title("Chat Bot")

        self.t = ""
                
        self.e = tk.Entry(self.root, bd=5, width=50)
        self.e.focus()
        self.e.bind(sequence='<Return>', func=self.user_response)
        self.e.pack(side="left")
        
        self.root.mainloop()

    #capture user response and call ai-response to get an answer?
    def user_response(self, event):
        #get text input by user and save it as self.t
        self.t = self.e.get()
        #clear event input text self.e for next user input
        self.e.delete(0, "end")
        #get answer text from the bot
        self.a = self.myCB.converse(self.t)
        self.w = tk.Label(self.root, text="Me: " + self.t, width=50)
        self.w.pack(fill="both", expand=True, padx=10, pady=10)

        self.ai_response(self.a)

    #test version only returns a random selected word as a response
    def ai_response(self,a):
        self.t = self.a
        self.w = tk.Label(self.root, text="AI: " + self.t, width=50)
        self.w.pack(fill="both", expand=True, padx=10, pady=10)


Chat()
