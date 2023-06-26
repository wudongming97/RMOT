import cv2
import os
import imageio

# img_root = '/data/wudongming/TransRMOT/exps/default/0020/light-color-cars-in-the-left/'
# fps = 30
# size = (1241, 376)
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# videowriter = cv2.VideoWriter('/data/wudongming/TransRMOT/0020.mp4', fourcc, fps, size)
# video_len = len(os.listdir(img_root)) - 2
# for i in range(0, video_len):
#     frame = cv2.imread(img_root + 'frame_{}.jpg'.format(str(i)))
#     cv2.putText(frame, 'Expression: light color cars in the left', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
#     videowriter.write(frame)
#
# videowriter.release()

path = '/data/wudongming/TransRMOT/exps/default/results_epoch99/0005/black-cars-in-the-left/'
gif_name = '/data/wudongming/TransRMOT/1.gif'
frames = []
pngFiles = os.listdir(path)
image_list = [os.path.join(path, f) for f in pngFiles if f.endswith('.jpg')]
for image_name in image_list[:300]:
    # 读取 png 图像文件
    frames.append(imageio.v2.imread(image_name))
# 保存为 gif
imageio.mimsave(gif_name, frames, fps=20)