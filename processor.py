import cv2
from matplotlib import pyplot as plt

for i in range(1, 6):
    im = cv2.imread(f"data/defect/{i}.jpeg", cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(im,100,255,cv2.THRESH_BINARY_INV)

    p = im.shape[1]//2
    cable = [i for i, v in enumerate(im[:, p]) if v > 0]
    dist = cable[-1] - cable[0]

    print(dist)
    
    plt.imshow(im, 'gray')
    plt.plot([p, p], [cable[0], cable[-1]], 'r')
    plt.show()
    