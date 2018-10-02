import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def transform(pos):
    pts = []
    n = len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))
    # print pts
    sums = {}
    diffs = {}
    tl = tr = bl = br = 0
    for i in pts:
        x = i[0]
        y = i[1]
        sum = x + y
        diff = y - x
        sums[sum] = i
        diffs[diff] = i
    sums = sorted(sums.items())
    diffs = sorted(diffs.items())
    n = len(sums)
    rect = [sums[0][1], diffs[0][1], diffs[n - 1][1], sums[n - 1][1]]
    #	   top-left   top-right   bottom-left   bottom-right

    h1 = np.sqrt((rect[0][0] - rect[2][0]) ** 2 + (rect[0][1] - rect[2][1]) ** 2)  # height of left side
    h2 = np.sqrt((rect[1][0] - rect[3][0]) ** 2 + (rect[1][1] - rect[3][1]) ** 2)  # height of right side
    h = max(h1, h2)

    w1 = np.sqrt((rect[0][0] - rect[1][0]) ** 2 + (rect[0][1] - rect[1][1]) ** 2)  # width of upper side
    w2 = np.sqrt((rect[2][0] - rect[3][0]) ** 2 + (rect[2][1] - rect[3][1]) ** 2)  # width of lower side
    w = max(w1, w2)

    # print '#',rect
    return int(w), int(h), rect


img = cv2.imread('A4Paper17.jpg')
r = 500.0 / img.shape[1]
dim = (500, int(img.shape[0] * r))

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#cv2.imshow('ORIGINAL', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11, 11), 0)
edge = cv2.Canny(gray, 100, 200)


rows = edge.shape[0]
cols = edge.shape[1]



_, contours, _ = cv2.findContours(edge.copy(), 1, 1)

n = len(contours)
max_area = 0
pos = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > max_area:
        max_area = area
        pos = i


peri = cv2.arcLength(pos, True)
approx = cv2.approxPolyDP(pos, 0.02 * peri, True)
size = img.shape


w, h, arr = transform(approx)
print('arr ={}'.format(arr))


pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
pts1 = np.float32(arr)
M = cv2.getPerspectiveTransform(pts1, pts2)
edge = cv2.warpPerspective(edge, M, (w, h))



plt.subplot(1,2,1)
plt.imshow(M)



print(w)
print(h)
rows = h
cols = w

edge[0:10,:]=0
edge[:,0:10]=0
edge[:,w-10:w]=0




print(len(np.unique(edge[42,:])))

top_toe_r=0

for r in range(10,rows):
    if len(np.unique(edge[r,:])) ==1:
        continue
    elif len(np.unique(edge[r,:])) >=2:
        top_toe_r = r
        foot_length = rows-r
        A4_length = rows
        ans = foot_length*29.7/A4_length
        accuracy = (24.8/ans)
        print('accuracy= {}'.format(accuracy))
        print('foot_length = {} cm'.format(ans))
        break


start = top_toe_r+30
end   = round(top_toe_r + (rows-top_toe_r)/2 + (rows-top_toe_r)/5)


max_c = 0
min_c = cols
for r in range(start,end):
    c = np.where(edge[r,:]>1)
    print('c = {}'.format(c))
    if c[0].shape[0]<=1:
        continue
    elif c[0].shape[0]>=2:
        print('c[0]={}'.format(c[0]))
        if c[0][0]<min_c:
            min_c = c[0][0]
        if c[0][-1]>max_c:
            max_c = c[0][-1]

print(max_c)
print(min_c)
if(cols != 0):
    print('width={}cm'.format(  (max_c-min_c)*21/cols   ))



plt.subplot(1,2,2)
plt.imshow(edge)
plt.show()

