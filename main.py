from detection_pipeline import detect
from utils import read_image, binarize, rgb2grey
from thresholding import otsu
import cv2

detect('o1.jpg')


# image = read_image("obrazy/o5.jpg")
# img = rgb2grey(image)
# img = binarize(img, otsu(img))
# cv2.imshow("Progowanie zwykłe", img)
# cv2.waitKey()
# cv2.imwrite("obrazy/otsu_o5.jpg", img)







