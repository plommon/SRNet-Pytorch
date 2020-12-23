import cv2

image_path = '../examples/gen_logs/iter-500000/002_o_t.png'
x = cv2.imread(image_path)
h, w, c = x.shape
filter_pixel = set([x[i][j][k] for i in range(h) for j in range(w) for k in range(c)])

# cv2.imshow('output', x)

# img_bright = cv2.convertScaleAbs(x, alpha=14.1667, beta=-1.7283e3)
img_bright = cv2.normalize(x, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imwrite('process.png', img_bright)
cv2.imshow('output', img_bright)
cv2.waitKey(0)
