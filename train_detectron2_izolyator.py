import copy
import cv2

import torch

import numpy as np
import matplotlib.pyplot as plt
import os, json
import random


from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T


def get_izolyator_dict(img_dir):
    fff = ['instances_default.json']
    dataset_dicts = []
    for name in fff:
        json_file = os.path.join(img_dir, name)
        with open(json_file) as f:
            imgs_anns = json.load(f)

        for idx, v in enumerate(imgs_anns['images']):
            record = {}

            filename = os.path.join(img_dir, v["file_name"])
            height, width = cv2.imread(filename).shape[:2]

            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            annos = imgs_anns['annotations']
            objs = []
            for anno in annos:
                if anno['image_id'] == v['id']:
                    obj = {
                        "bbox": anno['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": anno['segmentation'],
                        "category_id": 0,
                    }
                    objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts



class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is a custom version of the DatasetMapper. The only different with Detectron2's
    DatasetMapper is that we extract attributes from our dataset_dict.
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            # logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = [
                         T.RandomBrightness(0.5, 1.6),
                         T.RandomContrast(0.5, 1),
                         T.RandomSaturation(0.5, 1),
                         T.RandomRotation(angle=[-90, 90]),
                         T.RandomFlip(horizontal=True, vertical=False),
                         T.RandomCrop('relative_range', (0.4, 0.6)),
                         T.Resize((640,640)),
                         # CutOut()
                         ]

        # self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format = cfg.INPUT.FORMAT
        self.mask_on = cfg.MODEL.MASK_ON
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            # USER: Remove if you don't do semantic/panoptic segmentation.
        # if "sem_seg_file_name" in dataset_dict:
        #     with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
        #         sem_seg_gt = Image.open(f)
        #         sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
        #     sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
        #     sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
        #     dataset_dict["sem_seg"] = sem_seg_gt

        return dataset_dict


class izolyatorTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg))




def register_dataset(dataset_label, d):
    # Register dataset - if dataset is already registered, give it a new name
    try:
        DatasetCatalog.register(dataset_label, lambda d=d: get_izolyator_dict("dataset_bad_izolyator/" + d))
        MetadataCatalog.get(dataset_label).thing_classes = ['izolyator']
    except:
        # Add random int to dataset name to not run into 'Already registered' error
        n = random.randint(1, 1000)
        dataset_label = dataset_label + str(n)
        DatasetCatalog.register(dataset_label, lambda d=d: get_izolyator_dict("dataset_bad_izolyator/" + d))
        MetadataCatalog.get(dataset_label).thing_classes = ['izolyator']

    return MetadataCatalog.get(dataset_label), dataset_label

metadata, train_dataset = register_dataset('izolyator_train', "train")
# metadata, test_dataset = register_dataset('izolyator_test', "val")

izolyator_dict = get_izolyator_dict("dataset_bad_izolyator/train")

# for d in random.sample(izolyator_dict, 2):
#     plt.figure(figsize=(10,10))
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     plt.imshow(vis.get_image()[:, :, ::-1])
# plt.show()

MODEL_USE = 'faster_rcnn'
if MODEL_USE == 'faster_rcnn':
    MODEL_PATH = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    WEIGHT_PATH = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
elif MODEL_USE == 'ResNet':
    MODEL_PATH = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
    WEIGHT_PATH = 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'


def cfg_setup():
    cfg = get_cfg()
    cfg.OUTPUT_DIR = './out_izo'
    cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(WEIGHT_PATH)
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'
    cfg.SOLVER.BASE_LS = 0.005
    #     cfg.SOLVER.WARMUP_ITERS = 4500
    #     cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.MAX_ITER = 900
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg

cfg = cfg_setup()

trainer = izolyatorTrainer(cfg)

from detectron2.data import detection_utils as utils

train_data_loader = trainer.build_train_loader(cfg)

data_iter = iter(train_data_loader)

batch = next(data_iter)

plt.figure(figsize=(20, 20))

rows, cols = 2, 2
for i, per_image in enumerate(batch[:4]):
    plt.subplot(rows, cols, i + 1)

    # Pytorch tensor is in (C, H, W) format
    img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
    img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

    visualizer = Visualizer(img, metadata=metadata, scale=0.5)

    target_fields = per_image["instances"].get_fields()
    labels = None
    vis = visualizer.overlay_instances(
        labels=labels,
        boxes=target_fields.get("gt_boxes", None),
        masks=target_fields.get("gt_masks", None),
        keypoints=target_fields.get("gt_keypoints", None),
    )
    plt.imshow(vis.get_image()[:, :, ::-1])

plt.show()


trainer.resume_or_load(resume=False)

trainer.train()


#
#
# def cfg_test():
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(MODEL_PATH))
#     cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
#     cfg.DATASETS.TEST = (test_dataset,)
#     cfg.MODEL.RETINANET.NUM_CLASSES = 1
#     cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
#
#     return cfg
#
#
# cfg = cfg_test()
# predict = DefaultPredictor(cfg)
#
#
# def visual_predict(dataset):
#     for sample in dataset:
#         im = cv2.imread(sample['file_name'])
#         output = predict(im)
#
#         v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5)
#         v = v.draw_instance_predictions(output['instances'].to('cpu'))
#         plt.figure(figsize=(10, 10))
#         plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
#         plt.show()
#
#
# izolyator_test_dict = get_izolyator_dict("dataset_izolyator/val")
# visual_predict(izolyator_test_dict)