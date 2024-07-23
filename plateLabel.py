import cv2
import imageio
import numpy as np
import os
import shutil
import argparse
from alphabets import plate_chr

def allFileList(rootfile,allFile):
    """
    递归遍历指定目录，列出所有文件的路径。
    参数:
        rootfile: 根目录路径
        allFile: 存储文件路径的列表
    """
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)

def is_str_right(plate_name):
    """
    检查字符串中的所有字符是否都属于有效的车牌字符。
    参数:
        plate_name: 车牌号字符串
    返回:
        True 如果所有字符都有效，否则 False
    """
    for str_ in plate_name:
        if str_ not in palteStr:
            return False
    return True

if __name__=="__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/mnt/EPan/carPlate/@realTest2_noTraining/realrealTest", help='source') 
    parser.add_argument('--label_file', type=str, default='datasets/val.txt', help='model.pt path(s)')
    opt = parser.parse_args()

    rootPath = opt.image_path
    labelFile = opt.label_file
    palteStr=plate_chr  # 使用从alphabets.py导入的车牌字符
    print(len(palteStr))  # 输出字符集长度

    # plateDict ={}
    # for i in range(len(list(palteStr))):
    #     plateDict[palteStr[i]]=i
    plateDict = {char: idx for idx, char in enumerate(palteStr)}  # 创建字典将车牌字符映射到索引

    fp = open(labelFile,"w",encoding="utf-8")  # 打开文件写入标签数据
    file =[]
    allFileList(rootPath,file)  # 获取所有文件路径
    picNum = 0  # 记录处理的图片数量

    for jpgFile in file:
        print(jpgFile)
        jpgName = os.path.basename(jpgFile)
        name =jpgName.split("_")[0]  # 从文件名提取车牌号
        if " " in name:
            continue  # 如果名称中包含空格，跳过
        labelStr=" "
        if not is_str_right(name):
            continue  # 检查车牌号是否只包含有效字符

        strList = list(name)  # 将车牌号字符串转换为列表
        for  i in range(len(strList)):
            labelStr+=str(plateDict[strList[i]])+" "  # 将字符转换为对应的索引，并形成标签字符串
        # while i<7:
        #     labelStr+=str(0)+" "
        #     i+=1
        picNum+=1
        # print(jpgFile+labelStr)
        fp.write(jpgFile+labelStr+"\n")  # 写入文件路径和对应的标签

    fp.close()