import cv2

f = open("C:/Users/Robot1/Desktop/FODnew/Wrongpredictions.txt", 'r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    print(line)
    cv2.imshow(f""{line}"", f""{line}"")
    cv2.waitKey(0)