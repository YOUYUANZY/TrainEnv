import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from nets.facenet import Facenet as facenet
from utils.utils import preprocess_input, resize_image
from mtcnn import detect_faces


class FaceNet(object):
    def __init__(self, config):
        self.weightPath = config.weightPath
        self.inputSize = config.inputSize
        self.backbone = config.backbone
        self.resize = config.resize
        self.cuda = config.cuda
        self.net = None
        self.generate()

    def generate(self):
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.weightPath, map_location=device), strict=False)
        print('{} model loaded.'.format(self.weightPath))
        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def getFeature(self, image_1, image_2):
        bound, _ = detect_faces(image_1)
        if len(bound) == 0:
            raise ValueError('can\'t find face in image_1')
        image_1 = image_1.crop((bound[0][0], bound[0][1], bound[0][2], bound[0][3]))
        bound, _ = detect_faces(image_2)
        if len(bound) == 0:
            raise ValueError('can\'t find face in image_2')
        image_2 = image_2.crop((bound[0][0], bound[0][1], bound[0][2], bound[0][3]))
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.inputSize[1], self.inputSize[0]],
                                   letterbox_image=self.resize)
            image_2 = resize_image(image_2, [self.inputSize[1], self.inputSize[0]],
                                   letterbox_image=self.resize)
            photo_1 = torch.from_numpy(
                np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(
                np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()
            l1 = np.linalg.norm(output1 - output2, axis=1)

        print('Distance:%.6f' % l1)
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        #
        # plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va='bottom', fontsize=11)
        # plt.show()
        return output1, output2
