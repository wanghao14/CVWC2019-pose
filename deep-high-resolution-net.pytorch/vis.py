import os
import cv2
import json
import tqdm
import logging
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from collections import defaultdict
from pycocotools.coco import COCO

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

imgPath = './data/atrw/pose/images/test'
# annFile = 'data/atrw/pose/annotations/keypoints_val.json'
detecFile = './result/atrw/pose_hrnet/w48_384x288_OANW0730/results/keypoints_test_results_0.json'

# atrw = COCO(annFile)
atrw_detec = json.load(open(detecFile, 'r'))
detec_imgToAnn = defaultdict(list)
for anno in atrw_detec:
    detec_imgToAnn[anno['image_id']].append(anno['keypoints'])

# cats = [cat['name']
#         for cat in atrw.loadCats(atrw.getCatIds())]

# logger.info('=> class:{}'.format(cats))

# image_set_index = atrw.getImgIds()
_image_set_index = sorted(os.listdir('./data/atrw/pose/images/test'))
image_set_index = [int(x.split('.')[0]) for x in _image_set_index]
logger.info('=> num_images: {}'.format(len(image_set_index)))

for img_index in tqdm(image_set_index):
    img_name = '%06d.jpg' % int(img_index)
    img_path = os.path.join(imgPath, img_name)
    # logger.info('=> imgName: {}'.format(img_name))

    img = cv2.imread(img_path)
    # annIds = atrw.getAnnIds(imgIds=img_index, iscrowd=False)
    # objs = atrw.loadAnns(annIds)
    try:
        detecObj = detec_imgToAnn[img_index][0]
    except IndexError:
        logger.warning('!!Empty image_id: {}'.format(img_index))
        continue

    # unlabeled_num = 0
    for ipt in range(15):
        # if obj['keypoints'][ipt * 3 + 2] > 1:
        #     cv2.circle(img, (int(obj['keypoints'][ipt * 3 + 0]),
        #                      int(obj['keypoints'][ipt * 3 + 1])),
        #                4, [255, 0, 0], -1)
        #     cv2.putText(img, str(ipt + 1), (int(obj['keypoints'][ipt * 3 + 0]) + 1,
        #                                     int(obj['keypoints'][ipt * 3 + 1]) + 1),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255, 0, 0], 2)
        if detecObj[ipt * 3 + 2] > 0.25:
            cv2.circle(img, (int(detecObj[ipt * 3 + 0]),
                             int(detecObj[ipt * 3 + 1])),
                       4, [0, 0, 255], -1)
            cv2.putText(img, str(ipt + 1), ((int(detecObj[ipt * 3 + 0]) + 1,
                                             int(detecObj[ipt * 3 + 1]) + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, [0, 0, 255], 2)
    cv2.imwrite(os.path.join('./atrw_res/test', img_name), img)
    # cv2.imwrite(os.path.join(img_name), img)
# logger.info('=> the number of unlabeled picture: {}'.format(unlabeled_num))
