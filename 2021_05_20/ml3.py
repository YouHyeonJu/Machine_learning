import numpy as np
arr=np.arange(4)
print(arr)
print(arr.shape)
print("차원증가")
print()
print("행백터")
print("first")
row_vec1=arr.reshape(1,4)
print(row_vec1)
print(row_vec1.shape)

print("second")
row_vec=arr[np.newaxis,:]
print(row_vec)
print(row_vec.shape)

print("열백터")
print("first")
col_vec1=arr.reshape(4,1)
print(col_vec1)
print(col_vec1.shape)
print()
print("second")
col_vec=arr[:,np.newaxis]
print(col_vec)
print(col_vec.shape)
