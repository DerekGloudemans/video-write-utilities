import matplotlib.pyplot as plt
import PIL
import cv2

path = "C:\\Users\\derek\\Desktop\\cam_3.png"
im = PIL.Image.open(path)
plt.imshow(im)


path = "C:\\Users\\derek\\Desktop\\big map.jpg"
im = PIL.Image.open(path)
plt.figure()
plt.imshow(im)
