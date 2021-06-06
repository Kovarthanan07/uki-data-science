"""
@author: Kovarthanan K
"""

# using recursive function to calculate factorial
def factorial_Calc(n):
   if n == 1:
       return n
   else:
       return n*factorial_Calc(n-1)


def find_factorial():
    
    #asigning number to negative value for enable while loop
    number = -1 
    while(number < 0):
         number = int(input("Please input a nonnegative integer : ")) 
         
    factorial_output = 1
    
    if(number==0 or number==1):
        #when the input is 0 or 1
        print("Factorial of {} : {}".format(number, factorial_output))
    else : 
        #when input is more then 1   
        print("Factorial of {} : {}".format(number, factorial_Calc(number)))
        
#calling find_factorial function to find factorial
find_factorial()
