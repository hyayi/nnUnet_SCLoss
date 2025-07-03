import blosc2
import pickle

import numpy as np
from skimage.morphology import skeletonize
from skimage import measure
import scipy.ndimage as ndimage
import math
import numpy as np
from typing import List

import blosc2
import pickle

class OverlapTile:
    """
        This stategy is implementated from: Ma B, Ban X, Huang H, et al. Deep learning-based image segmentation
        for al-la alloy microscopic images[J]. Symmetry, 2018, 10(4): 107.
        For figure 4
        crop_size is the size of blue rectangle, which is equal to input_size
        roi_size is the size of yellow rectangle
    """

    def __init__(self, crop_size: int = 256, overlap_size: int = 32):
        self._crop_size = crop_size
        self._overlap_size = overlap_size
        self._roi_size = crop_size - 2 * overlap_size
        self._in_img_shape = None

    def crop(self, in_img: np.ndarray) -> List:
        """
            Crop in_img to sub-img List,
            hint: self._roi_size is used in crop and stitch stage, please check the paper careful when read this code
        """
        self._in_img_shape = in_img.shape
        # Pad image before cropping
        in_pad_img = np.pad(in_img, self._overlap_size, mode='symmetric')
        # calculate the number of cropping, which consider the influence of remainder
        h_pad_num = math.ceil(in_pad_img.shape[0] / self._roi_size)
        w_pad_num = math.ceil(in_pad_img.shape[1] / self._roi_size)
        # if in_pad_img.shape[0] % (self._roi_size) == 0:h_pad_num = h_pad_num - 1
        # if in_pad_img.shape[1] % (self._roi_size) == 0:w_pad_num = w_pad_num - 1
        in_crop_imgs = []
        for i in range(h_pad_num):
            # row analysis
            # overlap_crop, it is need to calculate the start of cropping
            start_h = i * self._roi_size
            end_h = start_h + self._crop_size

            # if there is some remainder result for cropping, change start_h and start_w
            if end_h > in_pad_img.shape[0]:
                start_h = in_pad_img.shape[0] - self._crop_size
                end_h = in_pad_img.shape[0]

            for j in range(w_pad_num):
                # column analysis
                start_w = j * self._roi_size
                end_w = start_w + self._crop_size

                if end_w > in_pad_img.shape[1]:
                    start_w = in_pad_img.shape[1] - self._crop_size
                    end_w = in_pad_img.shape[1]

                crop_img = in_pad_img[start_h: end_h, start_w: end_w]
                # print("cropping: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}, crop_shape={}".\
                #      format(i, start_h, start_w, j, end_h, end_w, crop_img.shape))
                in_crop_imgs.append(crop_img)
                if end_w == in_pad_img.shape[1]:
                    break
            if end_h == in_pad_img.shape[0]:
                break
        return in_crop_imgs

    def stitch(self, out_crop_imgs: List) -> np.ndarray:
        """
            Stitch sub-img List to whole out img
        """
        out_img = np.zeros(self._in_img_shape)

        # calculate the number of cropping, which consider the influence of remainder
        # h_num = math.ceil(self._in_img_shape[0] / self._roi_size) # it is no need to use that
        w_num = math.ceil(self._in_img_shape[1] / self._roi_size)

        for img_idx, out_crop_img in enumerate(out_crop_imgs):
            roi_img = out_crop_img[self._overlap_size: self._overlap_size + self._roi_size,
                      self._overlap_size: self._overlap_size + self._roi_size]
            i = int(img_idx / w_num)
            j = int(img_idx - (i * w_num))
            start_h = int(i * self._roi_size);
            end_h = start_h + self._roi_size
            start_w = int(j * self._roi_size);
            end_w = start_w + self._roi_size

            if end_h > self._in_img_shape[0]:
                start_h = self._in_img_shape[0] - self._roi_size
                end_h = self._in_img_shape[0]
            if end_w > self._in_img_shape[1]:
                start_w = self._in_img_shape[1] - self._roi_size
                end_w = self._in_img_shape[1]
                # print("stitching: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}"\
            # .format(i, start_h, start_w, j, end_h, end_w))
            out_img[start_h: end_h, start_w: end_w] = roi_img
        return out_img


class SkeletonAwareWeight():
    """
    Skeleton Aware Weight
    
    """
    def __init__(self, eps = 1e-20):
        """
        At this time, the weight is only suited to binary segmentation, so class_num = 2
        """
        self._class_num = 2
        self._eps = eps
        self.method=None
        self.overlap_tile = OverlapTile()  # speed up

    
    def _get_weight(self, mask: np.ndarray, method=None, single_border=False) -> np.ndarray:
        """
        Get skeleton aware weight map
        :param mask: binary gt mask with shape (H, W)
        :return weight:  weight map with shape (H, W, 2)
        """
        self.method = method
        # Get class weight for two channels
        weight = np.zeros((mask.shape[0], mask.shape[1], 2))
        class_weight = np.zeros((self._class_num, 1))
        for class_idx in range(self._class_num):
            idx_num = np.count_nonzero(mask == class_idx)
            class_weight[class_idx, 0] = idx_num
        min_num = np.amin(class_weight)
        class_weight = class_weight * 1.0 / min_num
        class_weight = np.sum(class_weight) - class_weight        
        
        
        # Get weight for each channel
        for class_idx in range(self._class_num):
            temp_mask = np.zeros_like(mask)
            temp_mask[mask == class_idx] = 1.0
            dis_trf = ndimage.distance_transform_edt(temp_mask)
            
            if class_idx == 1: 
                # Get weight for border
                if single_border:
                    temp_weight = 1.0
                else:
                    temp_weight = self._get_border_weight(class_weight[class_idx, 0], temp_mask, dis_trf)
            else:
                # Get weight for objects
                label_map, label_num = measure.label(temp_mask, connectivity=1, background=0, return_num=True)
                temp_weight = self._get_object_weight(class_weight[class_idx, 0], dis_trf, label_map, label_num)
            weight[:, :, class_idx] = temp_weight * temp_mask
        return weight
    
    def _get_border_weight(self, wc: float, mask: np.ndarray, dis_trf: np.ndarray) -> np.ndarray:
        """
        Get border weight for single connected object
        :param wc: class weight of border channel
        :param mask: real mask with shape (H,W), the border pixels equal 1 and the object pixels equal 0
        :param dis_trf: distance transform of border (shape (H,W)), it means the distance of each border pixel to the nearest object
        :return weight:  weight map of border with shape (H, W)        
        """

        sk = skeletonize(mask, method="lee") / 255  # Lee Skeleton method
        dis_trf_sk = dis_trf * sk   # Get the distance transform of skeleton pixel

        # Get the distance transform to skeleton pixel
        indices = np.zeros(((np.ndim(sk),) + sk.shape), dtype=np.int32)
        dis_trf_to_sk = ndimage.distance_transform_edt(1 - sk, return_indices=True, indices=indices)

        dis_sk_map = dis_trf_sk[indices[0, :, :], indices[1, :, :]] * mask

        max_dis_trf = np.amax(dis_trf_sk)  # min_dis[i, j, 0] == dis_trf

        weight = 2.0 - ((dis_sk_map + + self._eps) / (max_dis_trf + self._eps))

        weight[weight < 0] = 0.0
        return weight

    def _get_object_weight(self, wc: float, dis_trf: np.ndarray, label_map: np.ndarray, label_num: int) -> np.ndarray:
        """
        Get object weight
        :param wc: class weight of border channel
        :param dis_trf: distance transform of object (shape (H,W)), it means the distance of each pixel to the nearest border
        :param label_map: label map of connected components with shape (H, W)
        :param label_num: the number of connected components
        :return weight:  weight map of object with shape (H, W)        
        """
        weight = np.zeros(label_map.shape)
        image_props = measure.regionprops(label_map, cache=False)
        
        # For each connect component, calculate its weight by its skeleton
        for label_idx in range(label_num):
            image_prop = image_props[label_idx]
            (min_row, min_col, max_row, max_col) = image_prop.bbox
            bool_sub = np.zeros(image_prop.image.shape)
            bool_sub[image_prop.image] = 1.0

            bool_sub_sk = skeletonize(bool_sub, method="lee") /255 # Lee Skeleton method
            if np.count_nonzero(bool_sub_sk == 1.0) == 0:
                # If there is no skelenton pixel, continue
                continue
            # Get the distance transform of skeleton pixel
            dis_trf_sk_sub = dis_trf[min_row: max_row, min_col: max_col] * bool_sub_sk

            # Get the distance transform to skeleton pixel 
            indices = np.zeros(((np.ndim(bool_sub_sk),) + bool_sub_sk.shape), dtype=np.int32)
            dis_trf_to_sk = ndimage.distance_transform_edt(1-bool_sub_sk, return_indices=True, indices=indices)

            h, w = bool_sub.shape[:2]
            dis_sk_map = np.ones((h, w, 2))
            dis_sk_map[:, :, 0] = dis_trf_to_sk  # d0 
            dis_sk_map[:, :, 1] = dis_trf_sk_sub[indices[0, :, :], indices[1, :, :]]   # d1 

            # Rectify, enusre d0 <= d1, d0: the distance of pixel to nearest skeleton pixel, d1: the distance d1 of nearest skeleton pixel to border
            dis_sk_map[:, :, 0][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]] = dis_sk_map[:, :, 1][dis_sk_map[:, :, 0] > dis_sk_map[:, :, 1]]    
 
            weight_sub = 1-(dis_sk_map[:, :, 0] / (dis_sk_map[:, :, 1] + self._eps))
            
            weight[min_row: max_row, min_col: max_col] += weight_sub * bool_sub

        
        return weight

def load_data(data_path):
    if data_path.endswith('.npz'):
        # .npy 파일 처리
        data = np.load(data_path)['seg']
        return data

    elif data_path.endswith('.b2nd'):
        # .b2nd 파일 처리 (Blosc2 방식)
        dparams = {
            'nthreads': 1
        }
        data = blosc2.open(urlpath=data_path, mode='r', dparams=dparams, mmap_mode='r')
        return data
    else :
        raise ValueError(f"Unknown file type: {data_path}")

def load_pkl(path):
    with open(path, 'rb') as f:
        data =  pickle.load(f)

    return data
def save_pkl(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)