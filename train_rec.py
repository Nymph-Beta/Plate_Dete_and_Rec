import argparse
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
import yaml
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import lib.utils.utils as utils
from lib.dataset import get_dataset
from lib.core import function
from lib.utils.utils import model_info
from lib.models.plateNet import myNet_ocr
from alphabets import plateName, plate_chr
# from LPRNet import build_lprnet

from tensorboardX import SummaryWriter


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")

    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--img_h', type=int, default=48, help='height')
    parser.add_argument('--img_w', type=int, default=168, help='width')
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        # config = yaml.load(f, Loader=yaml.FullLoader)
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = plateName
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    config.HEIGHT = args.img_h
    config.WIDTH = args.img_w
    return config


def main():
    # load config
    config = parse_arg()

    # create output folder
    output_dict = utils.create_log_folder(config, phase='train')

    # cudnn
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # construct face related neural networks
    # cfg =[8,8,16,16,'M',32,32,'M',48,48,'M',64,128] #small model
    cfg = [16, 16, 32, 32, 'M', 64, 64, 'M', 96, 96, 'M', 128, 256]  # medium model
    # cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256] #big model
    # model = crnn.get_crnn(config,cfg=cfg)
    model = myNet_ocr(num_classes=len(plate_chr), cfg=cfg)
    # model = build_lprnet(num_classes=len(plate_chr))

    # get device
    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config.GPUID))
    else:
        device = torch.device("cpu:0")

    model = model.to(device)

    # define loss function
    criterion = torch.nn.CTCLoss()

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )

    if config.TRAIN.FINETUNE.IS_FINETUNE:
        model_state_file = config.TRAIN.FINETUNE.FINETUNE_CHECKPOINIT
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']

        # from collections import OrderedDict
        # model_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     if 'cnn' in k:
        #         model_dict[k[4:]] = v
        # model.cnn.load_state_dict(model_dict)
        model.load_state_dict(checkpoint)
        # if config.TRAIN.FINETUNE.FREEZE:
        #     for p in model.cnn.parameters():
        #         p.requires_grad = False

    elif config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file == '':
            print(" => no checkpoint found")
        checkpoint = torch.load(model_state_file, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            model.load_state_dict(checkpoint['state_dict'])
            last_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            model.load_state_dict(checkpoint)

    model_info(model)
    train_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = get_dataset(config)(config, input_w=config.WIDTH, input_h=config.HEIGHT, is_train=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.5
    converter = utils.strLabelConverter(config.DATASET.ALPHABETS)

    # 为新数据创建列表
    train_losses = []
    val_losses = []
    accuracies = []
    learning_rates = []

    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        # function.train(config, train_loader, train_dataset, converter, model,
        #                criterion, optimizer, device, epoch, writer_dict, output_dict)
        # lr_scheduler.step()
        #
        # acc = function.validate(config, val_loader, val_dataset, converter,
        #                         model, criterion, device, epoch, writer_dict, output_dict)

        train_info = function.train(config, train_loader, train_dataset, converter, model,
                                    criterion, optimizer, device, epoch, writer_dict, output_dict)
        val_info = function.validate(config, val_loader, val_dataset, converter,
                                     model, criterion, device, epoch, writer_dict, output_dict)

        # 单独获取accuracy和val_loss
        acc = val_info['accuracy']  # 获取准确率
        val_losses.append(val_info['val_loss'])
        accuracies.append(acc)

        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print("is best:", is_best)
        print("best acc is:", best_acc)

        # 收集数据
        train_losses.append(train_info['loss'])
        print(f"Epoch {epoch}: Added Training Loss: {train_info['loss']}")

        learning_rates.append(train_info['lr'])
        print(f"Epoch {epoch}: Added Learning Rate: {train_info['lr']}")

        # val_losses.append(val_info['val_loss'])
        print(f"Epoch {epoch}: Added Validation Loss: {val_info['val_loss']}")

        # accuracies.append(val_info['accuracy'])
        print(f"Epoch {epoch}: Added Accuracy: {val_info['accuracy']}")

        lr_scheduler.step()

        # save checkpoint
        torch.save(
            {
                "cfg": cfg,
                "state_dict": model.state_dict(),
                "epoch": epoch + 1,
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "best_acc": best_acc,
            }, os.path.join(output_dict['chs_dir'], "checkpoint_{}_acc_{:.4f}.pth".format(epoch, acc))
        )

    # print("Number of epochs:", config.TRAIN.END_EPOCH - last_epoch)
    # print("Length of train_losses:", len(train_losses))
    # print("Contents of train_losses:", train_losses)
    #
    # print("Length of val_losses:", len(val_losses))
    # print("Contents of val_losses:", val_losses)
    #
    # print("Length of accuracies:", len(accuracies))
    # print("Contents of accuracies:", accuracies)

    # 绘制图表
    # 绘制并保存损失和学习率图表

    # 假设已经有了以下数据列表
    epochs = range(1, len(train_losses) + 1)

    # 创建一个新的大图和子图
    plt.figure(figsize=(12, 10))

    # 第一个子图：训练损失
    plt.subplot(2, 2, 1)  # (行数, 列数, 索引)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 第二个子图：验证损失
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 第三个子图：学习率
    plt.subplot(2, 2, 3)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.legend()

    # 第四个子图：准确率
    plt.subplot(2, 2, 4)
    plt.plot(epochs, accuracies, label='Accuracy', color='purple')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # 调整子图间的间距
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(results_dir, 'training_summary.png'))
    plt.close()  # 关闭图形

    writer_dict['writer'].close()


if __name__ == '__main__':
    results_dir = 'rec_train_results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    main()
