## 该程序分为车牌检测和车牌识别两部分

### 车牌检测

ultralytics/datasets/yolov8-plate.yaml train和val路径是训练集和验证集的路径

运行yolo_train.py文件训练yolo模型，训练结果存储在runs文件夹中

### 车牌识别

将CCPD和CRPD截下来的车牌小图数据集，使用plateLable.py文件将数据集打上标签,生成train.txt和val.txt，存储在datasets文件夹中。

运行train_rec.py文件训练CRNN模型，训练结果存储在saved_model_of_rec文件夹中。

运行plate_rec_test.py文件测试识别效果，需要测试的图片存储在images文件夹中。


最后，运行detect_rec_plate.py文件实现完整的车牌检测和识别功能，需要测试文件存储在imgs文件夹中，输出结果存储在result文件夹中
