def divisible_by_k(n,k):
    a=0
    while n>0:
        if n%k==0:
            a=a+1
            print(n)
        n=n-1
    return a