# -*- coding: utf-8 -*-
"""
B351 - Project Chatbot
Test MySQL connection to hosted environment
"""

import MySQLdb
import configparser

def main():
    conf = configparser.ConfigParser()
    conf.read('db.ini')
    host = conf.get('Database','host')
    database = conf.get('Database','db')
    dbuser = conf.get('Database','user')
    dbpassword = conf.get('Database','password')
    db=MySQLdb.connect(host=host,db=database,user=dbuser,passwd=dbpassword)
    c=db.cursor()
    c.execute("""SELECT word FROM test_data""")
    result = c.fetchmany()
    print ("Words: "+str(result))

    
    
            
main()
    
