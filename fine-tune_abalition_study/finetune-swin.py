import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import torchvision.transforms.functional as TF
from PIL import Image
from torchsummary import summary
from torchvision import transforms
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix


import models.anynet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.5, 0.5, 1., 1.])
parser.add_argument('--datatype', default='2012',
                    help='datapath')
parser.add_argument('--datapath', default='/media/bsplab/62948A5B948A3219/data_scene_flow_2015/training/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=4,
                    
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=1,
                    help='batch size for training (default: 1)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 1)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
# parser.add_argument('--lr', type=float, default=5e-5,
#                     help='learning rate')
parser.add_argument('--lr', type=float, default=3e-5,#pir
                    help='learning rate')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--start_epoch_for_refine', type=int, default=0)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default='dataset/KITTI2015_val.txt')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--with_refine', action='store_true', help='with refine')


args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls


def main():
    global args
    df_mean = pd.DataFrame(index=range(1,args.epochs),columns=('epoch', 'stage1', 'stage2','stage3','stage4'))
    df_std = pd.DataFrame(index=range(1,args.epochs),columns=('epoch', 'stage1', 'stage2','stage3','stage4'))
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    # model1 = model.to(device)
    # summary(model1, input_size=[(3, 256, 512), (3, 256, 512)])
    # print(summary)
    model = nn.DataParallel(model).cuda()
    
   
    
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.Adagrad(model.parameters(), lr=args.lr, rho=0.9, eps=1e-6, weight_decay=0)

    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()
    if args.evaluate:
        test(TestImgLoader, model, log,df_mean,df_std)
        return

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        if epoch % 1 ==0:
            test(TestImgLoader, model, log,df_mean,df_std,epoch)

    test(TestImgLoader, model, log,df_mean,df_std)
#     potting of mean and sd
    # print(df_mean)
    # print(df_std)
    # plt.figure(1);plt.clf()
    # plt.plot(df_mean['epoch'],df_mean['stage1'],marker='o',color='blue')
    # plt.plot(df_mean['epoch'],df_mean['stage2'],marker='o',color='teal',linestyle='dashed')
    # plt.plot(df_mean['epoch'],df_mean['stage3'],marker='o',color='red')
    # plt.plot(df_mean['epoch'],df_mean['stage4'],marker='o',color='orange',linestyle='dashed')
    # plt.xlabel(r'epoch')
    # plt.ylabel(r'Mean')
    # plt.legend(['stage1','stage2','stage3','stage4'])
    # plt.savefig("img_mean.png")
    # plt.figure(2);plt.clf()
    # plt.plot(df_std['epoch'],df_std['stage1'],marker='o',color='blue')
    # plt.plot(df_std['epoch'],df_std['stage2'],marker='o',color='teal',linestyle='dashed')
    # plt.plot(df_std['epoch'],df_std['stage3'],marker='o',color='red')
    # plt.plot(df_std['epoch'],df_std['stage4'],marker='o',color='orange',linestyle='dashed')
    # plt.xlabel(r'epoch')
    # plt.ylabel(r'Standard_Deviation')
    # plt.legend(['stage1','stage2','stage3','stage4'])
    # plt.savefig("img_std.png")
    
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_refine
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)
    

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        # print(imgR.shape)
        # print(f'shape of left {imgL.shape}')
        model.cuda()
        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_refine:
            if epoch >= args.start_epoch_for_refine:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        # loss = [args.loss_weights[x] * F.smooth_l1_loss(reduction='mean')(outputs[x][mask], disp_L[mask])
        #         for x in range(num_out)]
        # mask=mask[0].view(1,256,512)
        # # disp_L = disp_L.view(1, 256, 512)
        # disp_L = torch.nn.functional.interpolate(disp_L.unsqueeze(0), size=(256, 512), mode='bilinear', align_corners=False).squeeze(0)
        # print(f'checkhere{outputs[x][mask].shape}')
        mask=mask[:,:,:511]
        disp_L=disp_L[:,:,:511]
        # print(disp_L.shape)
        # print(f'mask{mask.shape}')
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], reduction='mean') for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        if batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
            info_str = '\t'.join(info_str)

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)

average_of_stages=0
def test(dataloader, model, log,df_mean,df_std,epoch=0):
    cm_test  = np.zeros((10,10), dtype=int)
    stages = 3 + args.with_refine
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()
    
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()
        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16
            top_pad = (times+1)*16 - imgL.shape[2]
        else:
            top_pad = 0

        if imgL.shape[3] % 16 != 0:
            times = imgL.shape[3]//16
            right_pad = (times+1)*16-imgL.shape[3]
        else:
            right_pad = 0
        with torch.no_grad():
            # print(f'shape of left {imgL.shape}')
            outputs = model(imgL, imgR)
            for x in range(2):
                output = torch.squeeze(outputs[x], 1)
                if top_pad != 0:
                    output = output[:, top_pad:, :]                  
                    
                else:
                    output = output
                D1s[x].update(error_estimating(output, disp_L).item())
                
                
                _, yhat = torch.max(output.data, 1)
                yhat = yhat.cpu().numpy()
                y_true = np.argmax(disp_L.cpu().numpy(), axis=1)
                cm1=[]
                for i in range(8):
                    cm = multilabel_confusion_matrix(y_true[:,i], yhat[:,i])
                    cm1.append(cm)
                
                    
                fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 8))
                # print(fig,axes)
                for i, ax in enumerate(axes.flatten()):
                    if i < len(cm1):
                        cm = cm1[i]
                        im = ax.matshow(cm[0], cmap='Blues')
                        ax.set_title(f"Confusion matrix {i+1}")
                        ax.set_xlabel('Predicted label')
                        ax.set_ylabel('True label')
                        fig.colorbar(im, ax=ax)
                    else:
                        ax.axis('off')
                plt.savefig("img.png")
                plt.show()
                
        info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])
#         info_str_std = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].std) for x in range(stages)])
        
        
              

#         log.info('[{}/{}] {}'.format(
#             batch_idx, length_loader, info_str,info_str_std))

    
#     info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
#     info_str_std = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].std) for x in range(stages)])
    
    
#     log.info('Average test 3-Pixel Error mean= ' + info_str)
#     log.info('Average test 3-Pixel Error std= ' + info_str_std)
#     if epoch != 0:
#         for x in range(stages):
#             if x==0:
#                 stage1=D1s[x].avg
#                 stage1_std = D1s[x].std
#                 # df.loc['stage1'] =[D1s[x].avg]
#             if x==1:
#                 stage2=D1s[x].avg
#                 stage2_std = D1s[x].std
#                 # df.loc['stage2'] = [D1s[x].avg]
#             if x==2:
#                 stage3=D1s[x].avg
#                 stage3_std = D1s[x].std
#                 # df.loc['stage3'] = [D1s[x].avg]
#             if x==3:
#                 stage4=D1s[x].avg
#                 stage4_std = D1s[x].std
#                 # df.loc['stage4'] = [D1s[x].avg]
#         df_mean.loc[epoch] = [epoch,stage1,stage2,stage3,stage4]
#         df_std.loc[epoch] = [epoch,stage1_std,stage2_std,stage3_std,stage4_std]
    


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)
    
    
    disp = disp[:,:,:511]
    gt=gt[:,:,:511]
    errmap = torch.abs(disp - gt)
    mask=mask[:,:,:511]
    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.var= ((self.val - self.avg)**2)/(self.count)
        self.std = math.sqrt(self.var)

if __name__ == '__main__':
    main()

  