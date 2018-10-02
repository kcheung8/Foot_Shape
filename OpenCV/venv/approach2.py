import cv2
import numpy as np
import matplotlib.pyplot as plt
import timeit


def checkEqual2(iterator):
   return len(set(iterator)) <= 1

img1 = cv2.imread('approach2-1.jpg',1)
print(img1.shape[1])

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

#hsv = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)


low = np.array([100,100,100])
#low = np.array([200,200,200])
high = np.array([150,150,150])

image_mask = cv2.inRange(img1, low, high)
print(image_mask)


print('image_mask= {}'.format( type(image_mask)))
np.savetxt("foot.csv", image_mask, delimiter=",")
print(image_mask.shape)

plt.subplot(1,2,1)
plt.imshow(image_mask)

plt.subplot(1,2,2)
plt.imshow(img1)
plt.title("foot")
plt.show()



rows = image_mask.shape[0]
cols = image_mask.shape[1]




def function(image_mask):                                   #takes 7.9s
    for i in range(0,rows-200):
        for ii in range(0,cols):
            if (len(np.unique(image_mask[i:i+200,ii]))<2)  & (np.unique(image_mask[i:i+200,ii])[0]==0):
                print(' i and ii == {}   {}'.format(i,ii))
                return 'hi'


def function2(image_mask):                                  #takes 1.1s
    for r in range(0,rows-200):
        if len(np.unique(image_mask[r,:])) ==1:
            continue
        else:
            for c in range(0,cols):
                temp = np.unique(image_mask[r:r + 200, c])
                if (len(temp)<2)  & ((temp)[0]==255):
                    print(' r and c == {}   {}'.format(r,c))
                    return ((rows - r)/rows)*29.7


def function3(image_mask):                                  #takes 1.1s
    for r in range(0,rows):
        if len(np.unique(image_mask[r,:])) ==1:
            continue
        else:
            print('r == {}'.format(r))
            print('rows == {}'.format(rows))

            return ((rows-r)/rows) * 29.7



start = timeit.default_timer()

ans = function3(image_mask)
print("Predicted Foot Length: {} cm".format(ans))

stop = timeit.default_timer()
print('Run Time(second): ', stop - start )

print("Error {}cm".format(abs(ans -25.9)))
print('Accuracy: '+str(25.9/ans*100) +'%')