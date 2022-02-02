import numpy as np

a = np.array([1,2,3,1,1,1,3,4,5,6,4,3,2,2,3,4,3,2,23])
b = np.array([10,20,30,11,12,13,30,40,50,60,40,30,20,20,30,40,30,20,230])

sorted_index = np.argsort(-a, kind='stable')
a = a[sorted_index]
b = b[sorted_index]

u = np.unique(a)

class_list = []
for i in u:
    index = np.where(a==i)[0]
    print(index)
    class_list.append(b[index])

print("------------------------------------")
print("------------------------------------")
print(a)
print(b)
print("------------------------------------")
print(u)

print("------------------------------------")
print(class_list)
