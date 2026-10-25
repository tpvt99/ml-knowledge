#%%
import numpy as np
from numpy.lib.stride_tricks import as_strided

#%%
# follow https://medium.com/data-science/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20#6c04
# Number of strides is byte between element
#1/ slice the first 3
x = np.asarray(range(1,26), np.int8).reshape(5, 5)
print(x)
print(as_strided(x, shape=(3,), strides=(1,)))

#2/ slice the first 8
x = np.asarray(range(1,26), np.int8).reshape(5, 5)
print(x)
print(as_strided(x, shape=(8,), strides=(1,)))

#3/ flatten 2d array
x = np.asarray(range(1,26), np.int16).reshape(5, 5)
print(x.strides)
print(as_strided(x, shape=(25,), strides=(2,)))

# 4/ skip every other element
x = np.asarray(range(1, 26), dtype=np.int8).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,), strides=(2,)))

# 5/ stride the first column
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(4,), strides=(40,)))

# 6/ slice diagonal
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(5,), strides=(48,)))

# 7/ repeate the first element
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(5,), strides=(0,)))

# 8/ simple 2d slicing
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,4), strides=(40,8)))
# you can think as for axis=0, 3 elements, each element 40 bytes apart
# and for axis =1, 4 elements, each element is 8 bytes apart

# 9/ slicing a zigzag
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(4,2), strides=(48,8)))
# it is easy, for axis = 0, 4 elements, each element apart 48 bytes
# for axis=1, 2 elements, each element is only 8 bytes apart

# 10/ sparse slicing
x = np.asarray(range(1, 26), dtype=np.int64).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,3), strides=(10*8,2*8)))
# for axis=0, 1 to 11 is 10 elements away so 10*8 bytes
# for axis=1, 1 to 3 is 2 elements away so 2*8 bytes

# 11/ transpose 2d array
x = np.asarray(range(1, 26), dtype=np.int8).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(3,3), strides=(1,5*1)))
# for tranpose, axis=0, 1-2 is 1 element away so 1*1byte,
# for axis=1, 1-6 is 5 element away so 5*1 byte

#%%
# 12/ repeat the first column 4 times
x = np.asarray(range(1, 26), dtype=np.int32).reshape(5,5)
print(x.strides)
print(as_strided(x, shape=(5,4), strides=(5*4,0)))

# 13/ rehape 1d array to 2d array
x = np.asarray(range(1,13), np.int64)
print(x)
print(x.strides)
as_strided(x, shape=(4,3), strides=(3*8, 1*8))

# 14/ slide a 1d window
x = np.asarray(range(1,11), np.int8)
print(x.strides)
as_strided(x, shape=(8,3), strides=(1,1))

# 15/ slide a 2d window then flatten
x = np.asarray([0,1,10,11,20,21,30,31,40,41,50,51], np.int8).reshape(6,2)
print(x.strides)
as_strided(x, shape=(4,6), strides=(1*2,1))
# for axis=0 (each row) from 0-10 is 2 elements so 2*1byte = 2
# for axis=1 (each column) from 0-1, 1-10 is 1 element apart so 1*1 =1

# 16/ collapse an axis from 3d array
x = np.asarray(range(1,13), np.int8).reshape(3,2,2)
print(x.strides)
as_strided(x, shape=(3,4), strides=(4,1))
# %%
# 17/ 2 corners
x = np.asarray(range(1,26), np.int16).reshape(5,5)
as_strided(x, shape=(2,2,2), strides=(15*2, 5*2, 1*2))

# 18/ satggered slicing
x = np.asarray(range(1,26), np.int8).reshape(5,5)
as_strided(x, shape=(2,2,3), strides=(10*1, 6*1, 1*1))

# 19/ repeat a 2d array
x = np.asarray(range(1,26), np.int16).reshape(5,5)
as_strided(x, shape=(3,2,4), strides=(0*2, 5*2, 1*2))

# 20/ 3d transpose
x = np.asarray(range(1,13), np.int32).reshape(3,2,2)
as_strided(x, shape=(3,2,2), strides=(4*4, 1*4, 2*4))

# 21/ slide a 2d window
x = np.asarray(range(1,21), np.int64).reshape(4,5)
as_strided(x, shape=(3,2,5), strides=(5*8, 5*8, 1*8))

# 22/ reshape 1d array to 3d array
x = np.asarray(range(1,13), np.int8)
as_strided(x, shape=(2,2,3), strides=(6*1, 3*1, 1*1))

# %%
# 23/ slide a 2d receptive field for convolutions
x = np.asarray(range(1,26), np.int8)
as_strided(x, shape=(2,2,3,3), strides=(10*1, 2*1, 5*1, 1*1))

# 24/ repeat a 3d tensor
x = np.asarray(range(1,13), np.int64).reshape(2,2,3)
as_strided(x, shape=(2,2,2,3), strides=(6*8, 0*8, 3*8, 1*8))

# 25/ reshape 1d to 4d
x = np.asarray(range(1,17), np.int64)
as_strided(x, shape=(2,2,2,2), strides=(8*8, 4*8, 2*8, 1*8))


# %%
