'''
Copy and Paste from E2ETAD (https://github.com/xlliu7/E2E-TAD/blob/master/datasets/e2e_lib/videotransforms.py)
'''
import numpy as np
import numbers
import random
import cv2
from . import image_utils
import torch

class GroupRandomCrop(object):
    """Crop the given video sequences (t x h x w) at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        t, h, w, c = img.shape
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th) if h!=th else 0
        j = random.randint(0, w - tw) if w!=tw else 0
        return i, j, th, tw

    def __call__(self, imgs):
        
        i, j, h, w = self.get_params(imgs, self.size)

        imgs = imgs[:, i:i+h, j:j+w, :]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class GroupCenterCrop(object):
    """Crops the given seq Images at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        t, h, w, c = imgs.shape
        th, tw = self.size
        i = int(np.round((h - th) / 2.))
        j = int(np.round((w - tw) / 2.))

        return imgs[:, i:i+th, j:j+tw, :]


    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class GroupRandomHorizontalFlip(object):
    """Horizontally flip the given seq Images randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        if random.random() < self.p:
            # t x h x w
            return np.flip(imgs, axis=2).copy()
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class GroupResizeShorterSide(object):
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, imgs):
        '''n,h,w,c'''
        shorter = min(imgs[0].shape[:2])
        sc = self.new_size / shorter
        imgs = [cv2.resize(img, dsize=(0,0  ), fx=sc, fy=sc) for img in imgs]
        return np.asarray(imgs, dtype=np.float32)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.new_size)

class GroupPhotoMetricDistortion(object):
    """Apply photometric distortion to images sequentially, every
    transformation is applied with a probability of 0.5. The position of random
    contrast is in second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18,
                 p=0.5):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.p = p

    def __call__(self, imgs):
        """Call function to perform photometric distortion on images.
        Args:
            imgs (array): nhwc
        Returns:
            imgs (array): nhwc
        """
        
        assert imgs.dtype == np.float32, (
            'PhotoMetricDistortion needs the input imgs of dtype np.float32'
            ', please set "to_float32=True" in "LoadFrames" pipeline')

        def _filter(img):
            img[img < 0] = 0
            img[img > 255] = 255
            return img

        if np.random.uniform(0, 1) <= self.p:

            # random brightness
            if np.random.randint(2):
                delta = np.random.uniform(-self.brightness_delta,
                                       self.brightness_delta)
                imgs += delta
                imgs = _filter(imgs)

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = np.random.randint(2)
            if mode == 1:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # convert color from BGR to HSV
            imgs = np.array([image_utils.bgr2hsv(img) for img in imgs])

            # random saturation
            if np.random.randint(2):
                imgs[..., 1] *= np.random.uniform(self.saturation_lower,
                                               self.saturation_upper)

            # random hue
            # if np.random.randint(2):
            if True:
                imgs[..., 0] += np.random.uniform(-self.hue_delta, self.hue_delta)
                imgs[..., 0][imgs[..., 0] > 360] -= 360
                imgs[..., 0][imgs[..., 0] < 0] += 360

            # convert color from HSV to BGR
            imgs = np.array([image_utils.hsv2bgr(img) for img in imgs])
            imgs = _filter(imgs)

            # random contrast
            if mode == 0:
                if np.random.randint(2):
                    alpha = np.random.uniform(self.contrast_lower,
                                           self.contrast_upper)
                    imgs *= alpha
                    imgs = _filter(imgs)

            # randomly swap channels
            if np.random.randint(2):
                imgs = imgs[..., np.random.permutation(3)]

        return imgs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(\nbrightness_delta={self.brightness_delta},\n'
        repr_str += 'contrast_range='
        repr_str += f'{(self.contrast_lower, self.contrast_upper)},\n'
        repr_str += 'saturation_range='
        repr_str += f'{(self.saturation_lower, self.saturation_upper)},\n'
        repr_str += f'hue_delta={self.hue_delta})'
        return repr_str

class GroupRotate(object):
    """Spatially rotate images.
    Args:
        limit (int, list or tuple): Angle range, (min_angle, max_angle).
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            Default: bilinear
        border_mode (str): Border mode, accepted values are "constant",
            "isolated", "reflect", "reflect101", "replicate", "transparent",
            "wrap". Default: constant
        border_value (int): Border value. Default: 0
    """

    def __init__(self,
                 limit,
                 interpolation='bilinear',
                 border_mode='constant',
                 border_value=0,
                 p=0.5):
        if isinstance(limit, int):
            limit = (-limit, limit)
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.border_value = border_value
        self.p = p

    def __call__(self, imgs):
        """Call function to random rotate images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Spatially rotated results.
        """

        if np.random.uniform(0, 1) <= self.p:
            angle = np.random.uniform(*self.limit)
            
            imgs = [
                image_utils.imrotate(
                    img,
                    angle=angle,
                    interpolation=self.interpolation,
                    border_mode=self.border_mode,
                    border_value=self.border_value) for img in imgs
            ]
            imgs = np.array(imgs)

        return imgs

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(limit={self.limit},'
        repr_str += f'interpolation={self.interpolation},'
        repr_str += f'border_mode={self.border_mode},'
        repr_str += f'border_value={self.border_value},'
        repr_str += f'p={self.p})'

        return repr_str
