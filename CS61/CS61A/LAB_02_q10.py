def double_8(n):
    while n>0:
        if n%10==8:
            n=n//10
            if n%10==8:
                return True
        else:
            n=n//10
    return False