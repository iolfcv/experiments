import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import copy
import cv2
import json
import shutil
import os
import torchvision.models as models
import argparse
from torchvision.datasets import CIFAR10
from dataset_incr_crib import iDataset
from data_generator import shapenet_data_generator
from data_generator import toys_data_generator
from data_generator import all_classes
import torchvision.transforms as transforms



parser = argparse.ArgumentParser(description="Batch Learning")
parser.add_argument("--outfile", default="tmp", type=str, 
                    help="Output file name (without extension)")
parser.add_argument("--lr", default=0.01, type=float, 
                    help="Init learning rate")
parser.add_argument("--num_epoch", default=50, type=int, 
                    help="Number of epochs")
parser.add_argument("--batch_size", default=128, type=int, 
                    help="Mini batch size")
parser.add_argument("--img_size", default=224, type=int, 
                    help="Size of image to feed in the network")
parser.add_argument("--num_classes", default=20, type=int, 
                    help="Total number of classes")
parser.add_argument("--dataset", default="cifar", type=str, 
                    help="dataset for the batch learner")
parser.add_argument("--num_workers", default=8, type=int,
                    help="Number of worker threads for dataloader")
parser.add_argument("--optimizer", default="sgd", type=str,
                    help="Optimizer to use : adam/sgd")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M",
                    help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float,
                    metavar="W", help="weight decay (default: 1e-4)")
parser.add_argument("--step_size", default=30, type=int, 
                    help="Step size for learning rate decay scheduler")
parser.add_argument("--gamma", default=0.1, type=float, 
                    help="Decay factor for learning rate")
parser.add_argument("--resume", default="", type=str, metavar="PATH",
                    help="path to latest checkpoint, "
                         "without extension (default: none)")
parser.add_argument("--print_freq", default=1, type=int,
                    help="frequency of tqdm writes in an epoch")

# option for using CRIB
parser.add_argument("--h_ch", default=0.02, type=float,
                    help="Color jittering : max hue change")
parser.add_argument("--s_ch", default=0.05, type=float,
                    help="Color jittering : max saturation change")
parser.add_argument("--l_ch", default=0.1, type=float,
                    help="Color jittering : max lightness change")
parser.add_argument("--rendered_img_size", default=300, type=int,
                    help="Size of rendered images")
parser.add_argument("--lexp_len", default=100, type=int,
                    help="Number of frames in Learning Exposure")
parser.add_argument("--size_test", default=100, type=int,
                    help="Number of test images per object")
parser.add_argument("--num_repetitions", default=1, type=int,
                    help="Total number of repetitions of an instance (for TOYS)")
args = parser.parse_args()

torch.backends.cudnn.benchmark=True

if args.dataset == "cifar":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Data
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(args.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }

    class CIFAR100(CIFAR10):
        base_folder = "cifar-100-python"
        url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = "cifar-100-python.tar.gz"
        tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
        train_list = [
            ["train", "16019d7e3df5f24257cddd939b257f8d"],
        ]
        test_list = [
            ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
        ]

    trainset = CIFAR100(root="./data", train=True, download=True, 
                        transform=data_transforms["train"])
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=args.batch_size, 
                                              shuffle=True, 
                                              num_workers=args.num_workers,
                                              pin_memory=True)

    testset = CIFAR100(root="./data", train=False, download=True, 
                       transform=data_transforms["test"])
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=100, 
                                             shuffle=False, 
                                             num_workers=args.num_workers,
                                             pin_memory=True)

    # subset of train and test for <100 total classes
    classes = np.arange(args.num_classes)
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for cl in classes:
        train_data.append(trainset.train_data[cl == trainset.train_labels])
        train_labels.append(np.array(trainset.train_labels)[cl == trainset.train_labels])
        test_data.append(testset.test_data[cl == testset.test_labels])
        test_labels.append(np.array(testset.test_labels)[cl == testset.test_labels])
    trainset.train_data = np.concatenate(train_data, axis=0)
    trainset.train_labels = np.concatenate(train_labels,axis=0)
    testset.test_data = np.concatenate(test_data, axis=0)
    testset.test_labels = np.concatenate(test_labels, axis=0)

elif args.dataset == "toys":
    mean_image = np.load("data_generator/toys_mean_image.npy")
    mean_image.astype(np.uint8)
    mean_image = cv2.resize(mean_image, (args.rendered_img_size, 
                                         args.rendered_img_size))
    classes = np.array(all_classes.all_classes)
    classes = classes[:args.num_classes]
    class_map = {cl:i for i,cl in enumerate(classes)}

    # Passing multiple references to the same datagenerator for getting data
    # corresponding to repetitions
    train_dgs = [[toys_data_generator.DataGenerator(
                  model_name=cl, 
                  n_frames=args.lexp_len, 
                  size_test=args.size_test,
                  resolution=args.rendered_img_size)]*args.num_repetitions 
        for cl in classes]

    test_dgs = [[toys_data_generator.DataGenerator(
                 model_name=cl, 
                 n_frames=args.lexp_len, 
                 size_test=args.size_test,
                 resolution=args.rendered_img_size)] 
        for cl in classes]

    max_train_data_size = (2 * args.num_classes 
                           * args.lexp_len * args.num_repetitions)
    max_test_data_size = args.num_classes * args.size_test

    # Correct settings for initializing iDataset
    args.algo = "icarl"
    args.jitter = True
    
    print("Loading data...")
    trainset = iDataset(args, mean_image, train_dgs, max_train_data_size, 
                        classes, class_map, "batch_train")
    testset = iDataset(args, mean_image, test_dgs, max_test_data_size, 
                       classes, class_map, "test")
    print("Train set length:", len(trainset))
    print("Done")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


elif args.dataset == "shapenet":
    # HARDCODED
    num_instance_per_class = 25
    num_test_instance_per_class = 15

    mean_image = np.load("data_generator/shapenet_mean_image.npy")
    mean_image.astype(np.uint8)
    mean_image = cv2.resize(mean_image, (args.rendered_img_size, 
                                         args.rendered_img_size))

    train_instances = {}
    test_instances = {}

    with open("data_generator/shapenet_train_instances.json", "r") as tm_file:
        train_instances = json.load(tm_file)
        for cl in train_instances:
            tmp_list = []
            for synset, modelID in train_instances[cl]:
                tmp_list.append(modelID)
            
            train_instances[cl] = np.random.choice(tmp_list, 
                                                   num_instance_per_class,
                                                   replace=False)

    with open("data_generator/shapenet_test_instances.json") as tm_file:
        test_instances = json.load(tm_file)
        for cl in test_instances:
            tmp_list = []
            for synset, modelID in test_instances[cl]:
                tmp_list.append(modelID)
            test_instances[cl] = tmp_list

    classes = [cl for cl in train_instances]
    classes.sort() # So the order is fixed
    classes = classes[:args.num_classes]
    class_map = {cl:i for i,cl in enumerate(classes)}


    train_dgs = [[shapenet_data_generator.DataGenerator(
                category_name=cl, 
                instance_name=instance, 
                n_frames=args.lexp_len, 
                size_test=args.size_test,
                resolution=args.rendered_img_size, 
                job="train") 
            for instance in train_instances[cl]] 
        for cl in classes]

    test_dgs = [[shapenet_data_generator.DataGenerator(
                category_name=cl, 
                instance_name=instance, 
                n_frames=args.lexp_len, 
                size_test=args.size_test,
                resolution=args.rendered_img_size, 
                job="test") 
            for instance in test_instances[cl]] 
        for cl in classes]

    max_train_data_size = (2 * args.num_classes 
                           * args.lexp_len 
                           * num_instance_per_class)
    max_test_data_size = (args.num_classes 
                          * args.size_test 
                          * num_test_instance_per_class)

    # Correct settings for initializing iDataset
    args.algo = "icarl"
    args.jitter = True

    print("Loading data...")
    trainset = iDataset(args, mean_image, train_dgs, max_train_data_size, 
                        classes, class_map, "batch_train")
    testset = iDataset(args, mean_image, test_dgs, max_test_data_size, 
                       classes, class_map, "test")
    print("Done")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


# model
model = models.resnet34(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, args.num_classes, bias=False)
model = nn.DataParallel(model)
model.cuda()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
if args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                      step_size=args.step_size, 
                                      gamma=args.gamma)
args.start_epoch = 0
train_epoch_loss = []
test_accuracy = []
best_acc = 0

if args.resume:
    if os.path.isfile(args.resume+".pth.tar"):
        print("=> loading checkpoint {}".format(args.resume+".pth.tar"))
        checkpoint = torch.load(args.resume+".pth.tar")
        args.start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_acc = np.load(args.resume+".npz")
        train_epoch_loss = list(loss_acc["loss"])
        test_accuracy = list(loss_acc["test_accuracy"])
        print("=> loaded checkpoint {} (epoch {})"
              .format(args.resume+".pth.tar", checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at {}".format(args.resume+".pth.tar"))


def train(e, epoch):
    model.train()
    train_loss = 0.0
    with tqdm(total=len(trainloader)) as pbar:
        for i, data in enumerate(trainloader):
            if len(data) == 2:
                images, labels = data
            else:
                _, images, labels = data
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*len(labels)
            if (i+1) % args.print_freq == 0:
                tqdm.write("Epoch [%d/%d], Iter [%d/%d] Loss: %.4f" 
                    %(e+1, epoch, i+1, 
                      len(trainset)//args.batch_size, loss.item()))
            pbar.update(1)
    train_loss /= len(trainset)
    train_epoch_loss.append(train_loss)

def test():
    total = 0.0
    correct = 0.0
    model.eval()
    for i, data in enumerate(testloader):
        if len(data) == 2:
            images, labels = data
        else:
            _, images, labels = data
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        logits = model(images)
        _, pred = logits.max(1)
        total += len(labels)
        correct += (pred == labels).sum()
    accuracy = (float(correct)*100.)/total
    test_accuracy.append(accuracy)
    print("Test Accuracy: ", accuracy)


for e in range(args.start_epoch, args.num_epoch):
    if args.optimizer == "sgd":
        scheduler.step()
    
    train(e, args.num_epoch)
    test()

    is_best = False
    if test_accuracy[-1] > best_acc:
        best_acc = test_accuracy[-1]
        is_best = True

    torch.save({"epoch": e + 1,
                "state_dict": model.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                }, args.outfile + ".pth.tar")

    np.savez(args.outfile+".npz", loss=np.array(train_epoch_loss), 
             test_accuracy=np.array(test_accuracy))


