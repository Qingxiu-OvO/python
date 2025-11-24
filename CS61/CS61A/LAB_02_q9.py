def f(x):
    total=0
    while x>0:
        total,x=total+x%10,x//10
    return total