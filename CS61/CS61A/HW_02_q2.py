from operator import add, mul

def square(x):
    return x * x

def identity(x):
    return x

def triple(x):
    return 3 * x

def increment(x):
    return x + 1

def accumulate ( fuse , start , n , term ) :
    result = start
    count = 1

    if n == 0 :
        return result

    while n >= count :
        result = fuse ( result , term ( count ) ) 
        count = count + 1 

    return result