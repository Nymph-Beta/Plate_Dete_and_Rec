import torch
import cv2
import numpy as np
import argparse
import copy
import time
import os
from ultralytics.nn.tasks import  attempt_load_weights
from plate_recognition.plate_rec import get_plate_result,init_model,cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from fonts.cv_puttext import cv2ImgAddText

def allFilePath(rootPath,allFIleList):
    """
    递归读取指定目录下的所有文件，并将它们的路径添加到列表中。
    参数:
        rootPath: 要遍历的根目录路径
        allFIleList: 存储文件路径的列表
    """
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
            
def four_point_transform(image, pts):
    """
    对图像进行透视变换以提取车牌区域,得到车牌小图
    参数:
        image: 输入的OpenCV图像
        pts: 四个点坐标，用于透视变换的源点
    返回:
        warped: 变换后的图像
    """
    # rect = order_points(pts)
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
            

def letter_box(img,size=(640,640)):
    """
    yolo前处理,对图像进行letter box处理，调整图像大小并在周围填充，使其适合模型输入。
    参数:
        img: 输入的OpenCV图像
        size: 输出图像的期望大小
    返回:
        img: 调整大小并填充后的图像
        r: 缩放比例
        left, top: 填充的左边和顶部边界
    """
    h,w,_=img.shape
    r=min(size[0]/h,size[1]/w)
    new_h,new_w=int(h*r),int(w*r)
    new_img = cv2.resize(img,(new_w,new_h))
    left= int((size[1]-new_w)/2)
    top=int((size[0]-new_h)/2)   
    right = size[1]-left-new_w
    bottom=size[0]-top-new_h 
    img =cv2.copyMakeBorder(new_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def load_model(weights, device):  #加载yolov8 模型
    model = attempt_load_weights(weights,device=device)  # load FP32 model
    return model    

def xywh2xyxy(det):
    """
    将边界框的格式从中心点和宽高转换为左上角和右下角坐标, xywh转化为xyxy。
    参数:
        det: 检测到的边界框，格式为(x, y, w, h)
    返回:
        y: 转换后的边界框，格式为(x_min, y_min, x_max, y_max)
    """
    y = det.clone()
    y[:,0]=det[:,0]-det[0:,2]/2
    y[:,1]=det[:,1]-det[0:,3]/2
    y[:,2]=det[:,0]+det[0:,2]/2
    y[:,3]=det[:,1]+det[0:,3]/2
    return y

def my_nums(dets,iou_thresh):
    """
    非最大抑制(NMS)算法来消除冗余边界框。
    参数:
        dets: 检测到的边界框及其分数
        iou_thresh: IOU阈值用于决定何时丢弃框
    返回:
        keep: 经过NMS处理后保留下来的边界框的索引
    """
    y = dets.clone()
    y_box_score = y[:,:5]
    index = torch.argsort(y_box_score[:,-1],descending=True)
    keep = []
    while index.size()[0]>0:
        i = index[0].item()
        keep.append(i)
        x1=torch.maximum(y_box_score[i,0],y_box_score[index[1:],0])
        y1=torch.maximum(y_box_score[i,1],y_box_score[index[1:],1])
        x2=torch.minimum(y_box_score[i,2],y_box_score[index[1:],2])
        y2=torch.minimum(y_box_score[i,3],y_box_score[index[1:],3])
        zero_=torch.tensor(0).to(device)
        w=torch.maximum(zero_,x2-x1)
        h=torch.maximum(zero_,y2-y1)
        inter_area = w*h
        nuion_area1 =(y_box_score[i,2]-y_box_score[i,0])*(y_box_score[i,3]-y_box_score[i,1]) #计算交集
        union_area2 =(y_box_score[index[1:],2]-y_box_score[index[1:],0])*(y_box_score[index[1:],3]-y_box_score[index[1:],1])#计算并集

        iou = inter_area/(nuion_area1+union_area2-inter_area)#计算iou
        
        idx = torch.where(iou<=iou_thresh)[0]   #保留iou小于iou_thresh的
        index=index[idx+1]

    return keep


def restore_box(dets,r,left,top):
    """
    将经过letter box处理后的边界框坐标还原到原始图像的坐标系统。
    参数:
        dets: 边界框的坐标
        r: 缩放比例
        left, top: 图像填充的左边和顶部边界
    返回:
        dets: 转换后的边界框坐标
    """

    dets[:,[0,2]]=dets[:,[0,2]]-left
    dets[:,[1,3]]= dets[:,[1,3]]-top
    dets[:,:4]/=r
    # dets[:,5:13]/=r

    return dets
    # pass

def post_processing(prediction,conf,iou_thresh,r,left,top):
    """
    对模型的预测结果进行后处理，包括应用NMS和坐标转换。
    参数:
        prediction: 模型的原始预测结果
        conf: 置信度阈值
        iou_thresh: IOU阈值
        r: 缩放比例
        left, top: 图像填充的左边和顶部边界
    返回:
        x: 处理后的结果，包括更新的边界框和分类置信度
    """

    prediction = prediction.permute(0,2,1).squeeze(0)
    xc = prediction[:, 4:6].amax(1) > conf  #过滤掉小于conf的框
    x = prediction[xc]
    if not len(x):
        return []
    boxes = x[:,:4]  #框
    boxes = xywh2xyxy(boxes)  #中心点 宽高 变为 左上 右下两个点
    score,index = torch.max(x[:,4:6],dim=-1,keepdim=True)  #找出得分和所属类别
    x = torch.cat((boxes,score,x[:,6:14],index),dim=1)  #重新组合
    
    score = x[:,4]
    keep =my_nums(x,iou_thresh)
    x=x[keep]
    x=restore_box(x,r,left,top)
    return x

def pre_processing(img,opt,device):
    """
    准备图像以供模型输入，包括调整大小、填充和归一化。
    参数:
        img: 原始图像
        opt: 配置选项，包含模型输入尺寸等参数
        device: 设备类型（CPU或GPU）
    返回:
        img: 处理后的图像
        r: 缩放比例
        left, top: 填充的左边和顶部边界
    """
    img, r,left,top= letter_box(img,(opt.img_size,opt.img_size))
    # print(img.shape)
    img=img[:,:,::-1].transpose((2,0,1)).copy()  #bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img/255.0
    img =img.unsqueeze(0)
    return img ,r,left,top

def det_rec_plate(img,img_ori,detect_model,plate_rec_model):
    """
    对输入图像进行车牌检测和识别处理。
    参数:
        img: 输入图像用于检测
        img_ori: 原始图像用于提取车牌区域
        detect_model: 车牌检测模型
        plate_rec_model: 车牌识别模型
    返回:
        result_list: 包含车牌信息和检测结果的列表
    """
    result_list=[]
    img,r,left,top = pre_processing(img,opt,device)  #前处理
    predict = detect_model(img)[0]                   
    outputs=post_processing(predict,0.3,0.5,r,left,top) #后处理
    for output in outputs:
        result_dict={}
        output = output.squeeze().cpu().numpy().tolist()
        rect=output[:4]
        rect = [int(x) for x in rect]
        label = output[-1]
        roi_img = img_ori[rect[1]:rect[3],rect[0]:rect[2]]
        # land_marks=np.array(output[5:13],dtype='int64').reshape(4,2)
        # roi_img = four_point_transform(img_ori,land_marks)   #透视变换得到车牌小图
        if int(label):        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
            roi_img=get_split_merge(roi_img)
        plate_number,rec_prob,plate_color,color_conf=get_plate_result(roi_img,device,plate_rec_model,is_color=True)
        
        result_dict['plate_no']=plate_number   #车牌号
        result_dict['plate_color']=plate_color   #车牌颜色
        result_dict['rect']=rect                      #车牌roi区域
        result_dict['detect_conf']=output[4]              #检测区域得分
        # result_dict['landmarks']=land_marks.tolist() #车牌角点坐标
        # result_dict['rec_conf']=rec_prob   #每个字符的概率
        result_dict['roi_height']=roi_img.shape[0]  #车牌高度
        # result_dict['plate_color']=plate_color
        # if is_color:
        result_dict['color_conf']=color_conf    #颜色得分
        result_dict['plate_type']=int(label)   #单双层 0单层 1双层
        result_list.append(result_dict)
    return result_list


def draw_result(orgimg,dict_list,is_color=False):
    """
    在原始图像上绘制检测和识别结果。
    参数:
        orgimg: 原始图像
        dict_list: 包含检测结果的字典列表
        is_color: 是否显示颜色信息
    返回:
        orgimg: 标注了结果的图像
    """
    result_str =""
    for result in dict_list:
        rect_area = result['rect']
        
        x,y,w,h = rect_area[0],rect_area[1],rect_area[2]-rect_area[0],rect_area[3]-rect_area[1]
        padding_w = 0.05*w
        padding_h = 0.11*h
        rect_area[0]=max(0,int(x-padding_w))
        rect_area[1]=max(0,int(y-padding_h))
        rect_area[2]=min(orgimg.shape[1],int(rect_area[2]+padding_w))
        rect_area[3]=min(orgimg.shape[0],int(rect_area[3]+padding_h))

        height_area = result['roi_height']
        # landmarks=result['landmarks']
        result_p = result['plate_no']
        if result['plate_type']==0:#单层
            result_p+=" "+result['plate_color']
        else:                             #双层
            result_p+=" "+result['plate_color']+"双层"
        result_str+=result_p+" "
        # for i in range(4):  #关键点
        #     cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2) #画框
        
        labelSize = cv2.getTextSize(result_p,cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #获得字体的大小
        if rect_area[0]+labelSize[0][0]>orgimg.shape[1]:                 #防止显示的文字越界
            rect_area[0]=int(orgimg.shape[1]-labelSize[0][0])
        orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.2*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)#画文字框,背景白色
        
        if len(result)>=6:
            orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
            # orgimg=cv2ImgAddText(orgimg,result_p,rect_area[0]-height_area,rect_area[1]-height_area-10,(0,255,0),height_area)
               
    print(result_str)
    return orgimg


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'weights/yolov8s.pt', help='model.pt path(s)')  # yolov8检测模型路径
    parser.add_argument('--rec_model', type=str, default=r'weights/plate_rec_color.pth', help='model.pt path(s)')  # 车牌识别模型路径
    parser.add_argument('--image_path', type=str, default=r'imgs', help='source')  # 待识别图片的文件夹路径
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')  #yolov8 网络模型输入大小
    parser.add_argument('--output', type=str, default='result', help='source')  # 输出结果的文件夹路径

    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    opt = parser.parse_args()
    save_path = opt.output                

    if not os.path.exists(save_path): 
        os.mkdir(save_path)

    # 加载检测模型和车牌识别模型
    detect_model = load_model(opt.detect_model, device)  #初始化yolov8识别模型
    plate_rec_model=init_model(device,opt.rec_model,is_color=True)  #初始化识别模型

    # 输出模型参数量
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("yolov8 detect params: %.2fM,rec params: %.2fM" % (total/1e6,total_1/1e6))
    
    detect_model.eval() 
    # print(detect_model)

    # 获取所有待处理图片的路径
    file_list = []
    allFilePath(opt.image_path,file_list)

    count=0
    time_all = 0
    time_begin=time.time()

    # 遍历每张图片，进行车牌检测和识别
    for pic_ in file_list:
        print(count,pic_,end=" ")
        time_b = time.time()  # 记录开始时间

        img = cv2.imread(pic_)
        img_ori = copy.deepcopy(img)  # 创建原始图像的副本

        # 进行车牌检测和识别
        result_list=det_rec_plate(img,img_ori,detect_model,plate_rec_model)

        time_e=time.time()  # 记录结束时间
        ori_img=draw_result(img,result_list)  # 将识别结果绘制到图像上

        # 保存处理后的图片
        img_name = os.path.basename(pic_)  
        save_img_path = os.path.join(save_path,img_name)  #图片保存的路径
        cv2.imwrite(save_img_path, ori_img)

        # 计算处理时间并更新统计数据
        time_gap = time_e-time_b   #计算单个图片识别耗时
        if count:
            time_all+=time_gap 
        count+=1
        # print(result_list)

        # 输出总时间和平均处理时间
        total_time = time.time() - time_begin
        average_time = time_all / (len(file_list) - 1) if len(file_list) > 1 else 0
        print(f"sumTime time is {total_time} s, average pic time is {average_time} s")