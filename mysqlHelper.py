# -*- coding: utf-8 -*-
"""
B351 - Project Chatbot

Project Team:
Tiana Deckard
Hans Thieme
David Bickel

"""

import MySQLdb
import configparser

class mysqlHelper:
    conf = configparser.ConfigParser()
    host = ""
    database = ""
    dbuser = ""
    dbpassword = ""
    
    def __init__(self, dbfile):
        self.conf.read(dbfile)
        self.host = self.conf.get('Database','host')
        self.database = self.conf.get('Database','db')
        self.dbuser = self.conf.get('Database','user')
        self.dbpassword = self.conf.get('Database','password') 

    def getQueryResult(self,query):
        self.db = MySQLdb.connect(host=self.host,db=self.database,user=self.dbuser,passwd=self.dbpassword)
        self.c = self.db.cursor()
        #get words
        self.c.execute(query)
        result = list(self.c)
        return result
