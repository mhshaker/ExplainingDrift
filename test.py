a = [1,2,3]
b = [5,6,7]

for index,(i,j) in enumerate(zip(a,b)):
    print(i)
    print(j)
    print("------------------------------------ ", index)