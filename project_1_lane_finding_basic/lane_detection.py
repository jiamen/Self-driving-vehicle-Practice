
import numpy as np
import cv2
from Line import Line


# 感兴趣区域选择
def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from 'vertices'. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    # 定义一个和输入图像同样大小的全黑图像mask，这个mask也称掩膜
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    # 根据输入图像的通道数，忽略的像素点是多通道的白色，还是单通道的白色
    if len(img.shape) > 2:      # 3 channel
        channel_count = img.shape[2]        # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    # [vertices]中的点组成了多边形，将在多边形内的mask像素点保留
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    # 与mask做"与"操作，即仅留下多边形部分的图像
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image, mask


def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    img should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    Returns resulting blend image computed as follows:
        initial_img * α + img * β + λ
    """
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, α, img, β, λ)


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.
    :param line_candidates: liens from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """

    # separate candidate lines according to their slope     # 区分正斜率和负斜率
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slopes = np.median([l.slope for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slopes)), 0
    left_lane = Line(x1, y1, x2, y2)

    # interpolates biases adn slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    lane_right_bias  = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32( np.round((img_shape[0] - lane_right_bias) / lane_right_slope) ), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane


""" 得到车道线 """
def get_lane_lines(color_image, solid_lines=True):
    """
    This function take as input a color road frame and tries to infer the lane lines in the image.
    :param color_image: input frame
    :param solid_lines: if True, only selected lane lines are returned. If False, all candidate lines are returned.
    :return: list of (candidate) lane lines.
    """
    # resize to 960 x 540
    color_image = cv2.resize(color_image, (960, 540))

    # convert to grayscale
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # perform gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

    # perform edge detection   边缘提取 突出车道线，我们对灰度化后的图像做边缘处理
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)

    # perform hough transform  对提取到的线进行霍夫变换，找到最有可能的车道线
    detected_lines = hough_lines_detection(img=img_edge,
                                           rho=2,
                                           theta=np.pi/180,
                                           threshold=1,
                                           min_line_len=15,
                                           max_line_gap=5)                 # !!!!!!!!!

    # convert (x1, y1, x2, y2) tuples into Lines    把两个点转换为线
    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

    # if 'solid_lines' infer the two lane lines  如果“实线”推断出两条车道线
    if solid_lines:
        candidate_lines = []
        for line in detected_lines:
            # consider only lines with slope between 30 π(3.14)/6 and 60 (π(3.14)/3) degrees
            if 0.5 <= np.abs(line.slope) <= 2:
                candidate_lines.append(line)

        # interpolate lines candidates to find both lanes    插入候选线以找到两条车道
        lane_lines = compute_lane_from_candidates(candidate_lines, img_gray.shape)
    else:
        # if not solid_lines, just return the hough transform output
        lane_lines = detected_lines

    return lane_lines


def smoothen_over_time(lane_lines):
    """
    Smooth the lane line inference over a window of frames and returns the average lines.
    平滑帧窗口上的车道线推断并返回平均线。
    """
    avg_line_lt = np.zeros((len(lane_lines), 4))
    avg_line_rt = np.zeros((len(lane_lines), 4))

    for t in range(0, len(lane_lines)):
        avg_line_lt[t] += lane_lines[t][0].get_coords()
        avg_line_rt[t] += lane_lines[t][1].get_coords()

    return Line(*np.mean(avg_line_lt, axis=0)), Line(*np.mean(avg_line_rt, axis=0))


""" 这是main函数调用的主要函数 """
def color_frame_pipeline(frames, solid_lines=True, temporal_smoothing=True):
    """
    Entry point for lane detection pipeline. Takes as input a list of frames (RGB) and returns an image (RGB)
    with overlaid the inferred road lanes. Eventually, len(frames)==1 in the case of a single image.
    """
    is_videoclip = len(frames) > 0

    img_h, img_w = frames[0].shape[0], frames[0].shape[1]

    lane_lines = []
    for t in range(0, len(frames)):
        inferred_lanes = get_lane_lines(color_image=frames[t], solid_lines=solid_lines)
        lane_lines.append(inferred_lanes)               # 存放车道线

    if temporal_smoothing and solid_lines:
        lane_lines = smoothen_over_time(lane_lines)     # 返回车道平均线
    else:
        lane_lines = lane_lines[0]

    # prepare empty mask on which lines are drawn
    line_img = np.zeros(shape=(img_h, img_w))

    # draw lanes found
    for lane in lane_lines:
        lane.draw(line_img)

    # keep only region of interest by masking
    vertices = np.array([[(50, img_h),
                          (450, 310),
                          (490, 310),
                          (img_w-50, img_h)]],
                        dtype=np.int32)

    img_masked, _ = region_of_interest(line_img, vertices)

    # make blend on color image
    img_color = frames[-1] if is_videoclip else frames[0]
    img_blend = weighted_img(img_masked, img_color, α=0.8, β=1., λ=0.)

    return img_blend
