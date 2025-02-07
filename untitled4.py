
import numpy as np

ones_arr = np.ones((5,5),dtype=int)
ones_arr

ones_arr * 255

import matplotlib.pyplot as plt

from PIL import Image

horse_img=Image.open("/content/Horse.jpg")
horse_img

horse_arr=np.array(horse_img)
horse_arr

np.ndarray.flatten(horse_arr)

plt.imshow(horse_arr)

np.empty(2)

np.arange(4)

np.arange(2,9,2)

np.linspace(0,10,num=5)

arr = np.array([2,1,5,6,4,3,8,9])
arr

np.sort(arr)

a=np.array([1,2,3,4])
b=np.array([5,6,7,8])

np.concatenate((a,b))

x=np.array([[1,2,3],[4,5,6]])
y=np.array([[7,8,9],[10,11,12]])

np.concatenate((x,y),axis=1)

np.concatenate((x,y),axis=0)

x1=np.array([[1,2,3],
            [4,5,6]])

np.sum(x1,axis=0)

np.sum(x1,axis=1)

a2d=np.array([[1,2,3],
              [4,5,6]])

a2d.shape

a2d.ndim

a3d=np.array([[[1,2],[3,4]],
    [[5,6],[7,8]]
])

a3d.shape

a3d.ndim

np.random.randint(10,size=5)

ard=np.random.rand(2,3,4,5)
ard

ard.shape

ard.size

ard.ndim

np.sum(ard,axis=0)

np.sum(ard,axis=3)

ashape=np.arange(6)
ashape

bshape=ashape.reshape(3,2)
bshape

c=np.reshape(bshape,(2,3))
c

dshape=np.reshape(ashape,newshape=(1,6),order='C')
dshape

xc = np.arange(1,25).reshape(2,12)
xc

np.hsplit(xc,3)

np.hsplit(xc,(3,4))

a= np.array([[1,2,3,4],
            [5,6,7,8 ],[9,10,11,12]])
a

b=a.copy()
b

b[0]=99
b

a

b1=a.view()
b1

b2=a[0,:]
b2

b2[0]=99
b2

a

v2=a.copy()
v2

"""#Basic Array Operations"""

data =np.array([1,2])
data

ones=np.ones(2,dtype=int)
ones

data+ones

data-ones

data*data

data/ones

data/data

ars=np.array([1,2,3,4])
ars

ars.sum()

ars.max()

ars.mean()

ars.min()

arb=np.array([[1,1],[2,2]])
arb

arb.sum(axis=0)

arb.sum(axis=1)

ds=np.array([1.0,2.0])

ds * 1.6

hh=np.array([1,2,3])
hh

hh.max()

hh.min()

hh.sum()

np.arange(4,5)

np.zeros(5)

np.ones(4)

np.ones(4,dtype=int)

dt=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
dt

dt.shape

dt[0,1]

dt[1:3]

dt[0:2,0]

rng=np.random.default_rng()
rng.random(3)

rng.random((3,2))

rng.integers(5,size=(2,4))

uniqArr=([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])
uniqArr

np.unique(uniqArr)

uniqvalues=np.unique(uniqArr)
print(uniqvalues)

uniqvalues,indices_list=np.unique(uniqArr,return_index=True)
print(indices_list)

uniqvalues,occurrence_count=np.unique(uniqArr,return_counts=True)
print(occurrence_count)

a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
print(a_2d)

unique_rows = np.unique(a_2d)
print(unique_rows)

unique_rows = np.unique(a_2d,axis=0)
print(unique_rows)

unique_rows,indices,occurrence_count = np.unique(a_2d,axis=0,return_counts=True,return_index=True)
print(unique_rows)

print(indices)

print(occurrence_count)

a_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 2, 3, 4]])
print(a_2d)

unique_rows,indices,occurrence_count = np.unique(a_2d,axis=1,return_counts=True,return_index=True)
print(unique_rows)

print(indices)

print(occurrence_count)

print(a_2d)

unique_rows,indices,occurrence_count = np.unique(a_2d,return_counts=True,return_index=True)
print(unique_rows)

print(indices)

print(occurrence_count)

import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,10,100)
y=np.sin(x)

plt.plot(x,y)
plt.show()

plt.plot(x,y,marker='o')
plt.title('Sine Wave')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()