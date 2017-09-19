import cv2

files = []
with open('raw-images/image_list.txt', 'r') as file:
    for line in file:
        files.append(line.rstrip('\n'))
        img = cv2.imread('raw-images/' + files[-1])
        cv2.imwrite('raw-images-2/' + files[-1], img)
