import cv2
import random
from python.AIResult import *

#Generic paint color
colors = list()
while len(colors) < 100:
    # Randomly generate RGB color values
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)
    color = (b, g, r)
    # Check if the same color already exists
    if color not in colors:
        colors.append(color)


def draw_detect_results(img, results):
    '''
    Draw detection results
    :param img: src img
    :param results: detect results
    '''
    for r in results:
        cv2.rectangle(img, (r.box[0], r.box[1]), (r.box[0] + r.box[2], r.box[1] + r.box[3]), colors[r.class_id], 3)
        label_str = "id:" + str(r.class_id) + " " + str(round(r.score, 2))
        cv2.putText(img, label_str, (r.box[0], r.box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[r.class_id], 2)
    cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
    cv2.imshow("detect", img)
    cv2.waitKey(0)