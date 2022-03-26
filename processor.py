import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
do_plot = False

folders = ["data/flat", "data/notch"]

r = []
labels = ["damaged", "new"]
for folder in folders:
    result = []
    for i in range(1, 6):
        im = cv2.imread(f"{folder}/{i}.jpeg", cv2.IMREAD_GRAYSCALE)
        plt.imshow(im, 'gray')
        _,im = cv2.threshold(im,120,255,cv2.THRESH_BINARY_INV)

        edges = cv2.Canny(im,50,150,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi/180,200)

        datas = []
        for l in lines:
            datas.append([l[0][0], l[0][1]])
        kmeans.fit(datas)

        rhos = [0, 0]
        thetas = [0, 0]
        nums = [0, 0]
        for l, label in zip(datas, kmeans.labels_):
            rhos[label] += l[0]
            thetas[label] += l[1]
            nums[label] += 1
        for i in range(len(rhos)):
            rhos[i] /= nums[i]
            thetas[i] /= nums[i]

        dists = []
        for rho, theta in zip(rhos, thetas):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 - im.shape[1]*(-b))
            y1 = int(y0 - im.shape[1]*(a))

            dists.append(int(y0 - (im.shape[1]//2 - x0)*(a)))

            plt.plot([x0, x1], [y0, y1], 'r')
            #cv2.line(im,(x1,y1),(x2,y2),(0,0,255),2)
        #plt.imshow(edges)
        #plt.show()

        result.append(abs(dists[1] - dists[0]))

        print(abs(dists[1] - dists[0]))

        p = im.shape[1]//2
        #cable = [i for i, v in enumerate(im[:, p]) if v > 0]
        #dist = cable[-1] - cable[0]

        #print(dist)
        
        plt.imshow(im, 'gray')
        plt.plot([p, p], [dists[0], dists[1]], 'r')
        if do_plot:
            plt.show()
    r.append(result)
plt.clf()
print(r[0])
print(r[1])
my_dict = {labels[0]: r[0], labels[1]: r[1]}
fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys())
plt.ylabel("Line thickness [pixels]")
plt.show()
        