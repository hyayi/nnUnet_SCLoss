
import torch
import numpy as np
from skimage.morphology import skeletonize, dilation, square
from skimage import measure
import scipy.ndimage as ndimage
import os 
import math
from typing import List
from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
import blosc2
import glob
import pickle
from nnunetv2.create_weight_and_skeleton.utils import SkeletonAwareWeight,load_data,load_pkl,save_pkl

class GetSkeletonWeight:
    def __init__(self,single_border=False, dilation_k=3, do_tube=True):
        """
        - single_border: 경계에 고정값 weight 부여 여부
        - dilation_k: background skeleton의 팽창 크기
        - do_tube: foreground skeleton을 tube 형태로 확장할지 여부
        """
        self.single_border = single_border
        self.dilation_k = dilation_k
        self.do_tube = do_tube
        self.skeaw = SkeletonAwareWeight()

    def apply(self, data_dict, **params):
        # (1) Prepare segmentation
        seg_all = data_dict['segmentation']
        bin_seg = (seg_all > 0).astype(np.uint8)  # binary mask

        # (2) class weight 계산
        class_weight = self._get_class_weight(bin_seg[0])

        # (3) skeleton-aware weight 계산
        weight = self.skeaw._get_weight(bin_seg[0], method="skeaw", single_border=self.single_border)

        # (4) skeleton map (foreground + background)
        skelen = self._get_skeletons(bin_seg[0])

        # (5) 결과 저장
        data_dict["class_weight"] = class_weight
        data_dict["weight"] = weight.transpose((2, 0, 1))  # (C, H, W)
        data_dict["skelen"] = skelen.transpose((2, 0, 1))  # (C, H, W)

        return data_dict

    def _get_class_weight(self, mask: np.ndarray, class_num: int = 2) -> np.ndarray:
        class_weight = np.zeros((class_num, 1), dtype=np.float32)
        for class_idx in range(class_num):
            idx_num = np.count_nonzero(mask == class_idx)
            class_weight[class_idx, 0] = idx_num
        min_num = np.amin(class_weight) + 1e-6
        class_weight = class_weight / min_num
        class_weight = np.sum(class_weight) - class_weight
        return class_weight

    def _get_skeletons(self, mask: np.ndarray) -> np.ndarray:
        # foreground skeleton
        fore_skel = skeletonize(mask, method="lee").astype(np.uint8)
        if self.do_tube:
            fore_skel = dilation(dilation(fore_skel, square(2)), square(2))

        # background skeleton
        background = (1 - mask)
        skelen_bg = np.zeros_like(mask)
        labeled, num = measure.label(background, connectivity=1, return_num=True)
        props = measure.regionprops(labeled)

        for p in props:
            minr, minc, maxr, maxc = p.bbox
            submask = np.zeros_like(p.image, dtype=np.uint8)
            submask[p.image] = 1
            sk = skeletonize(submask, method="lee").astype(np.uint8)
            skelen_bg[minr:maxr, minc:maxc] += sk

        skelen_bg = dilation(skelen_bg, square(self.dilation_k))

        # combine
        skelens = np.stack([skelen_bg, fore_skel], axis=-1)  # shape (H, W, 2)
        return skelens
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(i, dataset_folder, weight_class):

    data = load_data(i)
    file_name = os.path.basename(i).split(".")[0].replace("_seg", "")
    pkl_file_path = os.path.join(dataset_folder, file_name + '.pkl')
    pkl_data = load_pkl(pkl_file_path)

    data_type = data.dtype
    data = (data[0] > 0).astype(data_type)
    data_dict = {"segmentation": data}

    result = weight_class.apply(data_dict)
    pkl_data["class_weight"] = result["class_weight"]
    pkl_data["weight"] = result["weight"]
    pkl_data["skelen"] = result["skelen"]

    save_pkl(pkl_data, pkl_file_path)

def get_weight(dataset_name_or_id, single_border, dilation_k, do_tube, num_workers=None):

    preprocessed_dataset_folder_base = os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
    seg_dir = os.path.join(preprocessed_dataset_folder_base, "nnUNetPlans_2d")
    data_list = glob.glob(os.path.join(seg_dir, "*_seg.*"))

    weight_class = GetSkeletonWeight(single_border=single_border, dilation_k=dilation_k, do_tube=do_tube)

    args = [(i, seg_dir, weight_class) for i in data_list]

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    with Pool(num_workers) as pool:
        list(tqdm(pool.starmap(process_file, args), total=len(data_list), desc="Generating weights"))

# def get_weight(dataset_name_or_id,single_border,dilation_k,do_tube):
#     preprocessed_dataset_folder_base = os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(dataset_name_or_id))
#     weight_class = GetSkeletonWeight()
#     data_list = glob.glob(os.path.join(preprocessed_dataset_folder_base,"nnUNetPlans_2d","*_seg.*"))

#     for i in data_list:
#         data = load_data(i)
#         file_name = os.path.basename(i).split(".")[0].replace("_seg","")
#         pkl_file_path = os.path.join(preprocessed_dataset_folder_base,"nnUNetPlans_2d",file_name+'.pkl')
#         pkl_data = load_pkl(pkl_file_path)

#         data_type = data.dtype
#         data = (data[0] > 0).astype(data_type)
#         data_dict = {"segmentation":data}
#         result = weight_class.apply(data_dict)
#         pkl_data["class_weight"] = result["class_weight"]
#         data_dict["weight"] = result["weight"]
#         data_dict["skelen"] = result["skelen"]
#         save_pkl(pkl_data,pkl_file_path)

def get_weightmap_entry():
    import argparse

    parser = argparse.ArgumentParser()

    # 필수 positional argument
    parser.add_argument('dataset_name_or_id', type=int,
                        help="Dataset name or ID to train with")

    parser.add_argument('--single_border', action='store_true', default=False,
                        help="Use single border during processing (default: False)")

    parser.add_argument('--dilation_k', type=int, default=3,
                        help="Dilation kernel size (default: 3)")

    parser.add_argument('--do_tube', default=True,
                        help="Whether to apply tube processing (default: True)")
    args, unrecognized_args =  parser.parse_known_args()

    get_weight(args.dataset_name_or_id, args.single_border, args.dilation_k, args.do_tube)

if __name__ =="__main__":
    get_weight(2001,False,3,True)