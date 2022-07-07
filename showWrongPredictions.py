import cv2

f = open("C:/Users/Robot1/Desktop/FODnew/Wrongpredictions.txt", 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    print(line)
    img = cv2.imread(f"{line}")
    cv2.imshow(f"{line}", img)
    cv2.waitKey(0)