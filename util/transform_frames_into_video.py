import cv2
import os


img_root = '/data/wudongming/TransRMOT/exps/default/0007/parking-cars/'
fps = 30
size = (1242, 375)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videowriter = cv2.VideoWriter('/data/wudongming/TransRMOT/0007.mp4', fourcc, fps, size)
video_len = len(os.listdir(img_root)) - 2
for i in range(0, video_len):
    frame = cv2.imread(img_root + 'frame_{}.jpg'.format(str(i)))
    cv2.putText(frame, 'Expression: parking cars', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
    videowriter.write(frame)

videowriter.release()