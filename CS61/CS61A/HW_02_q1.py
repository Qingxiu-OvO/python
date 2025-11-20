from operator import add, mul

def square(x):
    return x * x

def identity(x):
    return x

def triple(x):
    return 3 * x

def increment(x):
    return x + 1

def product ( n , term ) :
    
    if n == 0 :
    
        return 0 

    m = 1 

    while n > 0 :

        m = m * term ( n )
        n = n - 1 
    
    return m