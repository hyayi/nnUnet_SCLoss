from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from typing import List,Tuple
import random 
from skimage.restoration import inpaint_biharmonic
import numpy as np
from skimage.morphology import binary_dilation,disk
import torch
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel

def getRandomCoordinates(labelmap: np.array, num: int,
        size: List[int]) -> List[Tuple]:
    allcoors = np.where(labelmap)
    if len(allcoors[0]) == 0:
        return []

    min_size = np.array(size)/2
    max_size = np.array(labelmap.shape) - min_size - 1

    coors = []
    while len(coors) < num:
        idx = random.randint(0, len(allcoors[0])-1)
        arr_coor = np.array([allcoors[i][idx] for i in range(len(allcoors))])

        if (arr_coor >= min_size).all() and (arr_coor <= max_size).all():
            coors.append( tuple(arr_coor) )

    return coors

def getNumberOfHoles(holes):
    if isinstance(holes, int):
        return holes
    return random.randint(min(holes), max(holes))

def getSize(size, dims):
    if isinstance(size, list):
        if len(size) == dims:
            return size
        msg = f"size specified to be {size} but the data has {dims} dimensions."
        raise ValueError(msg)
    else:
        return [size for _ in range(dims)]

def coor2slices(coors, size: List[int]):
    slices = []
    isodd = [s%2 for s in size]

    for coor in coors:
        sl = [slice(max(c-s//2, 0), c+s//2+o) for c,s,o in zip(coor, size, isodd)]
        # Adding the channel dimension
        slices.append(tuple([slice(0, None)] + sl))
    return slices

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel

class CoLeTra:
    def __init__(self, mix_ratio: List, ws: int) -> None:
        self.mix_ratio = mix_ratio
        self.ws = ws

    def __call__(self, img:np.array, inpainted: np.array,
            slices: List) -> np.array:

        if self.mix_ratio[0] == -1 or self.mix_ratio[1] == -1:
            mx1 = np.random.random()
            mx2 = 1-mx1
        elif self.mix_ratio[0] == -2 or self.mix_ratio[1] == -2:
            mx2 = gkern(l=self.ws, sig=3)
            mx1 = 1 - mx2
        else:
            mx1 = self.mix_ratio[0]
            mx2 = self.mix_ratio[1]


        for sl in slices:
            img[sl] = img[sl]*mx1 + inpainted[sl]*mx2

        return img

class CoLeTraTransform(BasicTransform):
    def __init__(self,holes:List[int] | int
                 ,size: List[int],
                 fill_type:List[int]=[-2,-2]):
                super().__init__()
                self.holes = holes
                self.size = size
                self.fill_type = fill_type
                self.transform = CoLeTra(mix_ratio=fill_type, ws=size[0])
    
    def apply(self, data_dict, **params):
        #inpainting 
        # import pdb; pdb.set_trace()
        img = data_dict["image"].numpy()
        seg = (data_dict["segmentation"].numpy()[0] > 0).astype(np.int16)

        inpainting = self._apply_inpinting(img,seg)

        num_holes = getNumberOfHoles(self.holes)
        size = getSize(size=self.size, dims=len(seg.shape))
        coors = getRandomCoordinates(seg, num_holes, size)
        slices = coor2slices(coors, size)
        img = self.transform(img,inpainting,slices)
        img = torch.from_numpy(img)
        data_dict['image'] = img
        return data_dict

    def _apply_inpinting(self,img,mask):
        selem_disk = disk(3)
        for _ in range(3):
            mask = binary_dilation(mask,selem_disk)
        inpainting = inpaint_biharmonic(img,mask,channel_axis=0)
        return inpainting
            