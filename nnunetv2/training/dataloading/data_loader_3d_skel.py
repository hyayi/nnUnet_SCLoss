import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from skimage.morphology import skeletonize, dilation
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd

def calculate_tubed_skeleton(seg_all, do_tube):
    bin_seg = (seg_all > 0)
    seg_all_skel = np.zeros_like(bin_seg, dtype=np.int16)
    
    # Skeletonize
    for b in range(bin_seg.shape[0]):
        if np.sum(bin_seg[b, 0]) == 0:
            continue
        skel = skeletonize(bin_seg[b, 0])
        skel = (skel > 0).astype(np.int16)
        if do_tube:
            skel = dilation(dilation(skel))
        skel *= seg_all[b, 0].astype(np.int16)
        seg_all_skel[b, 0] = skel
    return seg_all_skel

class nnUNetDataLoader3DSkel(nnUNetDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack((seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None]))
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):

                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    skels = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                        skels.append(tmp['skel'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                        skel_all = [torch.stack([s[i] for s in skels]) for i in range(len(skels[0]))]
                    else:
                        seg_all = torch.stack(segs)
                        skel_all = torch.stack(skels)
                    del segs, images, skels

            return {'data': data_all, 'target': seg_all, 'skel': skel_all, 'keys': selected_keys}

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

