def is_prime ( n ) :
    t,a=1,0
    while n >= t :
        if n % t == 0 :
            a = a + 1 
        t = t + 1
    if a == 2 :
        return True
    else :
        return False