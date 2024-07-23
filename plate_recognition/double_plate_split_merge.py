import os
import cv2
import numpy as np
def get_split_merge(img):
    """
    分割图像的上下两部分并重新拼接
    参数:
        img: 输入的OpenCV图像，numpy数组格式
    返回:
        拼接后的新图像
    """
    h,w,c = img.shape # 获取图像的高度、宽度和通道数
    img_upper = img[0:int(5/12*h),:] # 分割出图像的上部，大约是整个高度的5/12
    img_lower = img[int(1/3*h):,:] # 分割出图像的下部，从大约1/3的高度开始到底部

    # 调整上部图像的大小，使其宽度和下部图像的宽度一致，高度也调整为下部图像的高度
    img_upper = cv2.resize(img_upper,(img_lower.shape[1],img_lower.shape[0]))

    # 水平堆叠上部和下部图像
    new_img = np.hstack((img_upper,img_lower))
    return new_img

if __name__=="__main__":
    img = cv2.imread("double_plate/tmp8078.png")
    new_img =get_split_merge(img)
    cv2.imwrite("double_plate/new.jpg",new_img)
