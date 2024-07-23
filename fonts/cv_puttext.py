import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    """
    在图片上添加中文文本
    参数:
        img: OpenCV图像，numpy数组格式
        text: 要添加的文本字符串
        left, top: 文本在图像上的起始坐标
        textColor: 文本颜色，默认为绿色
        textSize: 文本字体大小，默认为20
    返回:
        修改后的OpenCV图像
    """
    if (isinstance(img, np.ndarray)):  #判断是否OpenCV图片类型; img是否为numpy数组类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "fonts/platech.ttf", textSize, encoding="utf-8")  # 加载字体文件，用于支持中文
    draw.text((left, top), text, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    imgPath = "result.jpg"
    img = cv2.imread(imgPath)

    # 调用cv2ImgAddText函数在图像上添加文本
    saveImg = cv2ImgAddText(img, '中国加油！', 50, 100, (255, 0, 0), 50)

    cv2.imwrite('save.jpg',saveImg)

    # 下面的代码可以用来在窗口中显示图像，但在此例中被注释掉了
    # cv2.imshow('display', saveImg)
    # cv2.waitKey()  # 等待用户按键，然后继续执行
    # cv2.waitKey()