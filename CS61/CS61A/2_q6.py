def falling(n,k):
    a=0
    while k>1:
        k=k-1
        a=n*(n-1)
        n=n-1
    return a