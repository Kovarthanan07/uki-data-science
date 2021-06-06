"""
@author: Kovarthanan K
"""

import re 

def is_email_valid(email):
    pattern = re.compile("^[a-zA-Z0-9_or\-or\.]+@+[a-zA-Z0-9or\.]+$")
    result = pattern.findall(email)
    return(True if(result) else False)

def email_validation():
    email = ""
    while(not is_email_valid(email)):
        email = input("Please input your email address : ")
    email_details = email.split('@')
    print("email : {}, username : {}, host : {}".format(email, 
                                                        email_details[0],
                                                        email_details[1]))
    
email_validation()
