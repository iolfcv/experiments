import numpy as np
import torch
import torch.utils.data
import cv2
import random
import sys
import copy
from utils.color_jitter.jitter import jitter
from utils.get_samples import get_samples


class iDataset(torch.utils.data.Dataset):
    def __init__(self, args, mean_image, data_generators, max_data_size,
                 classes=None, class_map=None, job=None, le_idx=-1):
        """
        Args : 
            args : arguments from the argument parser
            mean_image : the mean image over the dataset
            data_generators : a list of len(classes) lists corresponding to each 
                              class, each of which contains a data_generator 
                              object for each instance of the class
            max_data_size : used to preallocate memory in RAM for loading images
            classes : list of names of classes
            class_map : map of class name to its label 
            job : "train", "test", "batch_train", "batch_val"
                  "batch_val" loads learning exposures but no augmentations
                  "batch_train" loads learning exposures without negative patches
            le_idx : if job is "train", equals the learning exposure index
        """
        self.job = job
        self.algo = args.algo
        self.jitter = args.jitter
        self.img_size = args.img_size

        self.datax = np.empty((max_data_size,3,
                               self.img_size,self.img_size),
                               dtype=np.uint8)
        # Stores the mean image cropped out at 
        # appropriate bounding boxes for each datax
        self.data_means = np.empty((max_data_size, 3, 
                                    self.img_size, self.img_size), 
                                    dtype=np.uint8)
        self.datay = np.empty(max_data_size, dtype=int)
        self.bb = np.empty((max_data_size, 4), dtype=int)
        # Data z stores which learning exposure an image comes from 
        # and its frame index; populated only when job=="train"
        self.dataz = np.empty((max_data_size, 2), dtype=int)
        self.curr_len = 0

        self.h_ch, self.s_ch, self.l_ch = args.h_ch, args.s_ch, args.l_ch

        # 3 x rendered_img_size x rendered_img_size
        self.mean_image = mean_image

        # Can also initialize an empty set and expand later
        if len(data_generators) > 0:
            assert len(data_generators) == len(classes), \
                "Number of datagenerators and classes passed is different"
            self.expand(args, data_generators, classes, class_map, job, le_idx)

    # To allow using the same memory locations for datax
    def pseudo_init(self, args, data_generators, classes=None,
                    class_map=None, job=None, le_idx=-1):
        self.curr_len = 0
        if len(data_generators) > 0:
            assert len(data_generators) == len(classes), \
                "Number of datagenerators and classes passed is different"
            self.expand(args, data_generators, classes, class_map, job, le_idx)

    def __getitem__(self, index):
        curr_img = np.float32(self.datax[index])
        curr_mean = np.float32(self.data_means[index])
        bb = self.bb[index]

        if self.job == "train" or self.job == "batch_train":
            if self.algo == "icarl" or self.algo == "lwf":
                # Augment : Color jittering
                if self.jitter:
                    jitter(curr_img, self.h_ch, self.s_ch, self.l_ch)
                    curr_img = (curr_img - curr_mean) / 255.

                # Augment : Random crops and horizontal flips
                random_cropped = np.zeros(curr_img.shape, dtype=np.float32)
                padded = np.pad(curr_img, ((0, 0), (4, 4), (4, 4)), 
                                mode="constant")
                crops = np.random.random_integers(0, high=8, size=(1, 2))
                if (np.random.randint(2) > 0):
                    random_cropped[:, :, :] = padded[:, 
                        crops[0, 0]:(crops[0, 0] + self.img_size), 
                        crops[0, 1]:(crops[0, 1] + self.img_size)]
                else:
                    random_cropped[:, :, :] = padded[:, 
                        crops[0, 0]:(crops[0, 0] + self.img_size), 
                        crops[0, 1]:(crops[0, 1] + self.img_size)][:, :, ::-1]

                curr_img = torch.FloatTensor(random_cropped)
            elif self.algo == "e2e":
                # Augment : 
                # For 3 choices - original image, brightness augmented 
                # and contrast augmented
                rand_num = np.random.randint(0, 3)
                if rand_num == 1:
                    # brightness
                    brightness = np.random.randint(-63, 64)
                    curr_img = np.clip(curr_img + brightness, 0, 255)
                elif rand_num == 2:
                    # contrast
                    # random from 0.2 to 1.8 inclusive
                    contrast = np.random.random() * (1.6+1e-16) + 0.2  
                    curr_img = np.clip((curr_img-curr_mean) * contrast 
                                        + curr_mean, 0, 255)

                curr_img = (curr_img - curr_mean)/255.
                    
                # Random choice whether to crop
                rand_num = np.random.randint(0, 2)
                if rand_num == 1:
                    crops = np.random.random_integers(0, high=8, size=2)
                    padded = np.pad(curr_img,((0,0),(4,4),(4,4)), 
                                    mode="constant")
                    curr_img = padded[:,
                                      crops[0]:(crops[0]+self.img_size),
                                      crops[1]:(crops[1]+self.img_size)] 

                # Random choice whether to mirror horizontally
                rand_num = np.random.randint(0, 2)
                if rand_num == 1:
                    curr_img = copy.deepcopy(curr_img[:, :, ::-1])

        elif self.job == "test" or self.job == "batch_val":
            curr_img = torch.FloatTensor((curr_img - curr_mean)/255.)

        target = self.datay[index]

        return index, curr_img, target

    def __len__(self):
        return self.curr_len

    def get_image_class(self, label):
        curr_data_y = self.datay[:self.curr_len]
        return (self.datax[:self.curr_len][curr_data_y == label],
                self.data_means[:self.curr_len][curr_data_y == label], 
                self.dataz[:self.curr_len][curr_data_y ==label], 
                self.bb[:self.curr_len][curr_data_y == label])

    def clear(self):
        self.curr_len = 0

    def append(self, images, labels, bb, frame_data):
        """
        Append dataset with images, labels, and other metadata
        """
        if self.curr_len + len(images) > len(self.datax):
            raise Exception("Dataset max length exceeded")

        self.datax[self.curr_len:self.curr_len + len(images)] = images
        self.datay[self.curr_len:self.curr_len + len(labels)] = labels
        self.bb[self.curr_len:self.curr_len + len(bb)] = bb
        self.dataz[self.curr_len:self.curr_len + len(frame_data)] = frame_data

        # get new data_means
        data_means = np.array([cv2.resize(self.mean_image[b[2]:b[3], b[0]:b[1]],
            (self.img_size, self.img_size)).transpose(2, 0, 1) for b in bb])
        self.data_means[self.curr_len:self.curr_len+len(data_means)] = data_means

        self.curr_len += len(images)

    def expand(self, args, data_generators,
               classes, class_map, job, le_idx=-1):
        """
        Call data generator to get images and append to dataset
        """
        for i, cl in enumerate(classes):
            for data_generator in data_generators[i]:
                if (job == "train" 
                        or self.job == "batch_train" 
                        or self.job == "batch_val"):
                    images, bboxes = data_generator.getLearningExposure()
                elif job == "test":
                    images, bboxes = data_generator.getRandomPoints()
                else:
                    raise Exception(
                        "Operation should be either 'test' or 'train'")

                # For E2E, get negatively labelled samples only in the
                # first learning exposure
                get_negatives = False
                if self.job == "train":
                    if (self.algo == "icarl" or self.algo == "lwf"
                            or (self.algo == "e2e" and le_idx == 0)):
                        get_negatives = True

                [datax, datay, bb] = get_samples(
                    images, bboxes, cl, self.img_size, class_map, get_negatives)

                # store mean image for new class
                data_means = np.array([cv2.resize(self.mean_image[b[2]:b[3], b[0]:b[1]],
                    (self.img_size, self.img_size)).transpose(2, 0, 1) for b in bb])

                if self.curr_len + len(datax) > len(self.datax):
                    raise Exception("Dataset max length exceeded")

                self.datax[self.curr_len:self.curr_len + len(datax)] = datax
                self.data_means[self.curr_len:self.curr_len+len(data_means)] = data_means
                self.datay[self.curr_len:self.curr_len + len(datay)] = datay
                self.bb[self.curr_len:self.curr_len + len(bb)] = bb
                self.dataz[self.curr_len:self.curr_len + len(datax), 0] = le_idx
                
                # numbering the frames that come in. Each +ve frame is followed 
                # by a -ve background image when get_negatives is True
                if get_negatives:
                    self.dataz[self.curr_len:self.curr_len +
                               len(datax), 1] = np.arange(len(datax)) // 2
                else:
                    self.dataz[self.curr_len:self.curr_len +
                               len(datax), 1] = np.arange(len(datax))
                self.curr_len += len(datax)
