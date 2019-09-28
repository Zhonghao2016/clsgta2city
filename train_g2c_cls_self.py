import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

from model.deeplab_multi import DeeplabMultiFeature
from model.discriminator import FCDiscriminator, FCDiscriminatorCLS
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset_label import cityscapesDataSetLabel

import pdb

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './data/gta5_deeplab'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/Cityscapes/data'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 120000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

TARGET = 'cityscapes'
SET = 'train'

LAMBDA_CLS_ADV = 0.00005
LABEL_DIRECTORY_TARGET = 'cityscapes_ssl1'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def amp_backward(loss, optimizer, retain_graph=False):
    if APEX_AVAILABLE:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(retain_graph=retain_graph)
    else:
        loss.backward(retain_graph=retain_graph)


def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMultiFeature(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D2 = FCDiscriminator(num_classes=args.num_classes).to(device)

    model_D2.train()
    model_D2.to(device)


    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    cityset = cityscapesDataSetLabel(args.data_dir_target, args.data_list_target,
                                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                                    crop_size=input_size_target,
                                    mean=IMG_MEAN,
                                    set=args.set, label_folder=LABEL_DIRECTORY_TARGET)
    targetloader = data.DataLoader(cityset,
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D2.zero_grad()

    # init cls D
    model_clsD = []
    optimizer_clsD = []
    for i in range(args.num_classes):
        model_temp = FCDiscriminatorCLS(num_classes=args.num_classes).to(device).train()
        optimizer_temp = optim.Adam(model_temp.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_temp.zero_grad()
        #model_temp, optimizer_temp = amp.initialize(
        #    model_temp, optimizer_temp, opt_level="O1", 
        #    keep_batchnorm_fp32=None, loss_scale="dynamic"
        #)
        model_temp, optimizer_temp = amp.initialize(
            model_temp, optimizer_temp, opt_level="O1", 
            keep_batchnorm_fp32=None, loss_scale="dynamic"
        )
        model_clsD.append(model_temp)
        optimizer_clsD.append(optimizer_temp)

    model, optimizer = amp.initialize(
        model, optimizer, opt_level="O1", 
        keep_batchnorm_fp32=None, loss_scale="dynamic"
    )

    model_D2, optimizer_D2 = amp.initialize(
        model_D2, optimizer_D2, opt_level="O1", 
        keep_batchnorm_fp32=None, loss_scale="dynamic"
    )

    bce_loss = torch.nn.BCEWithLogitsLoss()
    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    
    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):

        loss_seg_value2 = 0
        loss_adv_target_value2 = 0
        loss_D_value2 = 0
        loss_cls_adv = 0
        loss_cls_adv_value = 0
        loss_cls_D = 0
        loss_cls_D_value = 0
        loss_self_seg_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D2.zero_grad()
        adjust_learning_rate_D(optimizer_D2, i_iter)

        for i in range(args.num_classes):
            optimizer_clsD[i].zero_grad()
            adjust_learning_rate_D(optimizer_clsD[i], i_iter)

        for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D2.parameters():
                param.requires_grad = False

            for i in range(args.num_classes):
                for param in model_clsD[i].parameters():
                    param.requires_grad = False

            # train with source
            
            _, batch = trainloader_iter.__next__()

            images, labels, _, _ = batch
            images = images.to(device)
            labels = labels.long().to(device)

            _, pred2 = model(images)
            pred2 = interp(pred2)

            loss_seg2 = seg_loss(pred2, labels)
            loss = loss_seg2

            # proper normalization
            loss = loss / args.iter_size
            amp_backward(loss, optimizer)
            loss_seg_value2 += loss_seg2.item() / args.iter_size
            
            # train with target

            _, batch = targetloader_iter.__next__()
            images, target_labels, _, name = batch
            images = images.to(device)
            target_labels = target_labels.long().to(device)

            _, pred_target2 = model(images)
            pred_target2 = interp_target(pred_target2)

            pred_target_score = F.softmax(pred_target2, dim=1)
            D_out2 = model_D2(pred_target_score)
            loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss_self_seg = seg_loss(pred_target2, target_labels)
            loss_self_seg = loss_self_seg / args.iter_size
            loss_self_seg_value = loss_self_seg.item()
            
            _, target_pred_cls = torch.max(pred_target_score, dim=1)
            target_pred_cls = target_pred_cls.long().detach()
            for i in range(args.num_classes):
                cls_mask = (target_pred_cls==i) * (target_labels==i)
                if torch.sum(cls_mask) == 0:
                    continue
                cls_gt = torch.tensor(target_labels.data).long().to(device)
                cls_gt[~cls_mask] = 255
                cls_gt[cls_mask] = source_label
                cls_out = model_clsD[i](pred_target_score)
                loss_cls_adv += seg_loss(cls_out, cls_gt)
            loss_cls_adv_value = loss_cls_adv.item() / args.iter_size
                    

            loss = args.lambda_adv_target2 * loss_adv_target2 + LAMBDA_CLS_ADV * loss_cls_adv + loss_self_seg
            loss = loss / args.iter_size
            amp_backward(loss, optimizer)
            loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size


            # train D

            # bring back requires_grad

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred2 = pred2.detach()
            D_out2 = model_D2(F.softmax(pred2, dim=1))

            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))
            loss_D2 = loss_D2 / args.iter_size / 2
            amp_backward(loss_D2, optimizer_D2)
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target2 = pred_target2.detach()
            D_out2 = model_D2(F.softmax(pred_target2, dim=1))

            loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))
            loss_D2 = loss_D2 / args.iter_size / 2
            amp_backward(loss_D2, optimizer_D2)
            loss_D_value2 += loss_D2.item()

            for i in range(args.num_classes):
                for param in model_clsD[i].parameters():
                    param.requires_grad = True

            pred_source_score = F.softmax(pred2, dim=1)
            _, source_pred_cls = torch.max(pred_source_score, dim=1)
            source_pred_cls = source_pred_cls.long().detach()
            for i in range(args.num_classes):
                cls_mask = (source_pred_cls==i) * (labels==i)
                if torch.sum(cls_mask) == 0:
                    continue
                cls_gt = torch.tensor(source_pred_cls.data).long().to(device)
                cls_gt[~cls_mask] = 255
                cls_gt[cls_mask] = source_label
                cls_out = model_clsD[i](pred_source_score)
                loss_cls_D = seg_loss(cls_out, cls_gt) / 2
                amp_backward(loss_cls_D, optimizer_clsD[i])
                loss_cls_D_value += loss_cls_D.item()

            pred_target_score = F.softmax(pred_target2, dim=1)
            _, target_pred_cls = torch.max(pred_target_score, dim=1)
            target_pred_cls = target_pred_cls.long().detach()
            for i in range(args.num_classes):
                cls_mask = (target_pred_cls==i) * (target_labels==i)
                if torch.sum(cls_mask) == 0:
                    continue
                cls_gt = torch.tensor(target_pred_cls.data).long().to(device)
                cls_gt[~cls_mask] = 255
                cls_gt[cls_mask] = target_label
                cls_out = model_clsD[i](pred_target_score)
                loss_cls_adv += seg_loss(cls_out, cls_gt)
                loss_cls_D = seg_loss(cls_out, cls_gt) / 2
                amp_backward(loss_cls_D, optimizer_clsD[i])
                loss_cls_D_value += loss_cls_D.item()



        optimizer.step()
        optimizer_D2.step()
        for i in range(args.num_classes):
            optimizer_clsD[i].step()

        if args.tensorboard:
            scalar_info = {
                'loss_seg2': loss_seg_value2,
                'loss_adv_target2': loss_adv_target_value2,
                'loss_D2': loss_D_value2,
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
        'iter = {0:8d}/{1:8d}, loss_seg2 = {2:.3f}, loss_adv2 = {3:.3f} loss_D2 = {4:.3f} loss_cls_adv = {5:.3f} loss_cls_D = {6:.3f} loss_self_seg = {7:.3f}'.format(
            i_iter, args.num_steps, loss_seg_value2, loss_adv_target_value2, loss_D_value2, loss_cls_adv_value, loss_cls_D_value, loss_self_seg_value))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
            for i in range(args.num_classes):
                torch.save(model_clsD[i].state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(NUM_STEPS) + '_clsD.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))
            for i in range(args.num_classes):
                torch.save(model_clsD[i].state_dict(), osp.join(args.snapshot_dir, 'GTA5_clsD'+str(i)+'.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
