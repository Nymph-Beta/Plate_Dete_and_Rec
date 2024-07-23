from plate_recognition.plateNet import myNet_ocr,myNet_ocr_color
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import sys

# 车牌识别系统，利用深度学习模型来识别图片中的车牌号码和颜色

def cv_imread(path):  #可以读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def allFilePath(rootPath,allFIleList):  # 递归遍历指定目录下的所有图片文件
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            # 判断是否是文件，以及文件类型是否是图片
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
color=['黑色','蓝色','绿色','白色','黄色']    
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value,std_value=(0.588,0.193)  # 图像归一化时使用的均值和标准差

def decodePlate(preds):  # 解析模型的预测结果，提取车牌号码
    pre=0
    newPreds=[]
    index=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
            index.append(i)
        pre=preds[i]
    return newPreds,index

def image_processing(img,device):    # 图像预处理函数，用于模型输入
    img = cv2.resize(img, (168,48))  # 调整图像大小
    img = np.reshape(img, (48, 168, 3))  # 重新调整图像形状

    # normalize归一化处理
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])  # 转置图像，以匹配模型输入格式
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())  # 调整图像tensor的形状以匹配批次大小
    return img

def get_plate_result(img,device,model,is_color=False):    # 主要的车牌识别函数
    input = image_processing(img,device)
    if is_color:  #是否识别颜色
        preds,color_preds = model(input)  # 获取车牌号码和颜色的预测结果
        color_preds = torch.softmax(color_preds,dim=-1)  # 应用softmax函数来计算颜色概率
        color_conf,color_index = torch.max(color_preds,dim=-1)  # 获取最可能的颜色和其置信度
        color_conf=color_conf.item()
    else:
        preds = model(input)  # 获取车牌号码的预测结果

    preds=torch.softmax(preds,dim=-1)
    prob,index=preds.max(dim=-1)  # 获取最可能的车牌字符和其概率
    index = index.view(-1).detach().cpu().numpy()  # 转换预测结果为numpy数组
    prob=prob.view(-1).detach().cpu().numpy()
   
    
    # preds=preds.view(-1).detach().cpu().numpy()
    newPreds,new_index=decodePlate(index)  # 解析预测结果，获取车牌号码
    prob=prob[new_index]  # 更新概率数组，仅包括车牌号码对应的字符
    plate=""
    for i in newPreds:
        plate+=plateName[i]
    # if not (plate[0] in plateName[1:44] ):
    #     return ""
    if is_color:
        return plate,prob,color[color_index],color_conf    #返回车牌号以及每个字符的概率,以及颜色，和颜色的概率
    else:
        return plate,prob

def init_model(device,model_path,is_color = False):    # 初始化模型
    # print( print(sys.path))
    # model_path ="plate_recognition/model/checkpoint_61_acc_0.9715.pth"
    check_point = torch.load(model_path,map_location=device)
    model_state=check_point['state_dict']  # 获取模型状态
    cfg=check_point['cfg']  # 获取模型配置
    color_classes=0
    if is_color:
        color_classes=5           #颜色类别数
    model = myNet_ocr_color(num_classes=len(plateName),export=True,cfg=cfg,color_num=color_classes)
   
    model.load_state_dict(model_state,strict=False)  # 加载模型权重
    model.to(device)
    model.eval()
    return model

# model = init_model(device)
if __name__ == '__main__':
   model_path = r"weights/plate_rec_color.pth"
   image_path ="images/tmp2424.png"
   testPath = r"/mnt/Gpan/Mydata/pytorchPorject/CRNN/crnn_plate_recognition/images"
   fileList=[]
   allFilePath(testPath,fileList)
#    result = get_plate_result(image_path,device)
#    print(result)
   is_color = False
   model = init_model(device,model_path,is_color=is_color)
   right=0
   begin = time.time()
   
   for imge_path in fileList:
        img=cv2.imread(imge_path)
        if is_color:
            plate,_,plate_color,_=get_plate_result(img,device,model,is_color=is_color)  # 识别车牌和颜色
            print(plate)
        else:
            plate,_=get_plate_result(img,device,model,is_color=is_color)  # 仅识别车牌
            print(plate,imge_path)
        
  
        
