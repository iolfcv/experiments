import sys
import os
import math
import numpy as np
import random
import cv2
import time


class DataGenerator(object):
    def __init__(self, n_frames, size_test, resolution, model_name):
        self.total_exposures = max([int(i) for i in next(
            os.walk('render_data/' + str(model_name)))[1]]) + 1
        self.bboxes = []
        for i in range(self.total_exposures):
            bboxes = np.load(
                os.path.join(
                    'render_data',
                    str(model_name),
                    str(i),
                    'bboxes.npy'))
            self.bboxes.append(bboxes)

        # bboxes for random view_points generated
        self.test_bboxes = np.load(
            os.path.join(
                'test_data',
                str(model_name),
                'bboxes.npy'))

        self.n_frames = n_frames
        self.size_test = size_test
        self.resolution = resolution
        self.model_name = model_name
        self.n_exposures = 0

    def info(self):
        outStr = ' n_frames: %d \n resolution: %d \n model_name: %s \n'\
                 ' n_exposures: %d \n total_exposures: %d \n ' % (
                    self.n_frames, self.resolution, self.model_name, 
                    self.n_exposures, self.total_exposures)
        print(outStr)

    def getLearningExposure(self):
        '''
        Returns a learning exposure of n_frames frames of RGB images, 
        their respective bounding boxes and the percentage covered
        '''
        images, bboxes = self.get_img_bboxes('learning_exposure')
        self.n_exposures += 1
        return images, bboxes

    def getRandomPoints(self):
        '''
        Returns random points of length self.n_frames for testing
        '''
        images, bboxes = self.get_img_bboxes('random')
        return images, bboxes

    def get_img_bboxes(self, kind):

        if kind == 'random':
            amount = self.size_test
        elif kind == 'learning_exposure':
            amount = self.n_frames

        images = np.zeros((self.n_frames, self.resolution, self.resolution, 3))

        for frame in range(amount):
            img, _ = self.read_img(frame, kind)

            # flipping BGR to RGB
            img = img[:, :, [2, 1, 0]]
            images[frame] = img

        if kind == 'random':
            bboxes = self.test_bboxes
        elif kind == 'learning_exposure':
            bboxes = self.bboxes[self.n_exposures]

        return (images, bboxes.astype(np.int32))

    def read_img(self, frame, kind):

        if kind == 'random':
            im_filepath = 'test_data/' + self.model_name + \
                '/' + ('%04d' % frame) + '.png'
        elif kind == 'learning_exposure':
            im_filepath = 'render_data/' + self.model_name + '/' + \
                str(self.n_exposures) + '/' + ('%04d' % frame) + '.png'

        try:
            img = cv2.imread(im_filepath, cv2.IMREAD_UNCHANGED)
        except Exception as _err:
            sys.stderr.write(
                'Something went wrong reading file: %s' % im_filepath)
            raise Exception(_err)

        if not isinstance(img, np.ndarray):
            print('Something wrong with file: %s' % im_filepath)
            raise Exception('File Corrupted')

        return img, im_filepath