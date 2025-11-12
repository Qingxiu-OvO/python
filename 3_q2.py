def fizzbuzz(n):
    t=0
    while n>0:
        t,n=1+t,n-1
        if t%3==0 and t%5==0:
            print('fizzbuzz')
        elif t%3==0 :
                print('fizz')
        elif t%5==0 :
            print('buzz')
        else: 
            print(t)