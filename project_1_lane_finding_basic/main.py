
import matplotlib.pyplot as plt
import cv2
import os
from os.path import join, basename
from collections import deque
from lane_detection import color_frame_pipeline



if __name__ == '__main__':
    resize_h, resize_w = 540, 960

    verbose = True
    if verbose:
        plt.ion()                                           # 打开交互模式
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()                   # 窗口最大化，此时窗口内还没有内容
        # figManager.frame.Maximize(True)


    # 第一次测试：test on images
    test_images_dir = join('data', 'test_images')       # 文件夹路径
    test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]     # 加载所有图片，形成测试数组

    for test_img in test_images:
        print('Processing image: {}'.format(test_img))

        out_path  = join('out', 'images', basename(test_img))
        in_image  = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        out_image = color_frame_pipeline([in_image], solid_lines=True)                      # △△△  重中之重  ☆☆☆
        cv2.imwrite(out_path, cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR))                   # 保存找到车道线并标记车道线的图片
        if verbose:
            plt.imshow(out_image)
            plt.waitforbuttonpress()
    plt.close('all')


    # 第二次测试：test on videos
    test_videos_dir = join('data', 'test_videos')
    test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]

    for test_video in test_videos:
        print('Processing video: {}'.format(test_video))

        cap = cv2.VideoCapture(test_video)
        # VideoCapture()中参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频，如vt = cv2.VideoCapture("../testi.mp4")
        out = cv2.VideoWriter(join('out', 'videos', basename(test_video)),  # 连接，得到名称
                              fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                              fps=20.0, frameSize=(resize_w, resize_h))
        # fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数，注意：字符顺序不能弄混
        # cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi
        # cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi
        # cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi
        # cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv
        # cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv


        frame_buffer = deque(maxlen=10)

        while cap.isOpened():
            ret, color_frame = cap.read()
            
            if ret:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
                color_frame = cv2.resize(color_frame, (resize_w, resize_h))
                frame_buffer.append(color_frame)
                blend_frame = color_frame_pipeline(frames=frame_buffer, solid_lines=True, temporal_smoothing=True)      # △△△  重中之重  ☆☆☆
                out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))        # blend: 使混合; 掺和; (和某物)混合; 融合; (使)调和，协调;
                cv2.imshow('blend', cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR)), cv2.waitKey(1)
            else:
                break

        cap.release()                   # 释放视频流
        out.release()
        cv2.destroyAllWindows()         # 关闭所有窗口



