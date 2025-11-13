def has_digit( n , k ) : 
    while n > 0 :
        if n % 10 == k :
            return True
        n = n // 10 
    return False

def unique_digits ( n ) :
    k , m = 0 , 0
    while k <= 9 :
        if has_digit ( n , k ) == True :
            m = m + 1
        k = k + 1
    print ( m ) 