import cv2

img = cv2.imread("task4_handout/food/food/lena.png")

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
imgCanny = cv2.Canny(img,150,200)

cv2.imshow("Image", img)
cv2.imshow("Gray image", imgGray)
cv2.imshow("Image Blurred", imgBlur)
cv2.imshow("Image Canny", imgCanny)


cv2.waitKey(0)
