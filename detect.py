from __future__ import division

from utils.utils import *
from utils.datasets import *
import os
import os.path as osp
import time
import datetime
import cv2
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable



def detect_save(model, data_path, img_size,batch_size,epoch_cur,store_path='test_result',thres=[0.7,0.1]):

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    conf_thres = thres[0]
    nms_thres = thres[1]

    if os.path.exists(store_path):

        store_path_file = osp.join(store_path, str(epoch_cur))
    else:
        print("Store path not exist, build the file:", store_path)
        os.mkdir(store_path)

    model.eval()  # Set in evaluation mode
    dataloader = DataLoader(
        ImageFolder(data_path, img_size=img_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # class_name_list
    classes = ['Nodule']
    print("\nPerforming object detection:")
    prev_time = time.time()


    if len(os.listdir(store_path)) > 0:
        raise UserWarning('The result path is not empty!This may lead to ambiguity.' )

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)


    # print("\nSaving result:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        img = cv2.imread(path)
        if detections is not None:
            detections = rescale_boxes(detections, img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 150), 1)
                cv2.putText(img, classes[int(cls_pred)]+' conf: '+str(cls_conf.item()), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 150, 150), 1)
        # save images:
        image_name = path.split("/")[-1]


        image_path = osp.join(store_path, image_name)
        print("Save result: ",image_path)
        print('#'*10)
        cv2.imwrite(image_path, img)





