cimport cv2
import matplotlib.pyplot as plt
import numpy as np
#video = cv2.VideoCapture(0)
#if video.isOpened():
#    while True:
#        check, frame = video.read()
#        cv2.circle(frame, (155,120),5,(0,0,255),-1)
#        cv2.circle(frame, (480,120),5,(0,0,255),-1)
#        cv2.circle(frame, (155,120),5,(0,0,255),-1)
#        cv2.circle(frame, (155,120),5,(0,0,255),-1)

#        if check:
#            cv2.imshow('Color Frame', frame)
#            key = cv2.waitKey(50)
#            if key == ord('q'):
#                break
#        else:
#            print('Frame not available')
#            print(video.isOpened())


img = cv2.imread('A4Paper2.jpg')

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)



#253,129
#1067,284
#65,1175
#870,1340

cv2.circle(img, (842,1146),10,(0,0,255),-1)
cv2.circle(img, (2704,1158),10,(0,0,255),-1)
cv2.circle(img, (191,2118),10,(0,0,255),-1)
cv2.circle(img, (2833,2294),10,(0,0,255),-1)


pts1 = np.float32([[842,1146],[2704,1158],[191,2118],[2833,2294]])
pts2 = np.float32([[0,0],[400,0],[0,600],[400,600]])

matrix = cv2.getPerspectiveTransform(pts1,pts2)
result = cv2.warpPerspective(img, matrix, (400,600))

pts3 = np.float32([[0,0],[1,0],[0,1],[1,1]])
pts4 = np.float32([[3,3],[4,3],[3,4],[4,4]])
matrix2 = cv2.getPerspectiveTransform(pts3,pts4)
print(pts3)
print(pts4)


print('matrix2 = {}'.format(matrix2))

plt.subplot(1,2,1)
plt.imshow(img)

plt.subplot(1,2,2)
plt.imshow(result)

plt.show()
