# Yolov3-for-thyroid-nodule-detection
﻿# YOLOV3 for thyroid nodules detection

We trained Yolov3 on the data set of 1805 b-ultrasound images of thyroid nodules. And the weights are stored in  ./pretrained_model .The test interface  is offered here.


# Introductions


The forward process of Yolov3 will run on the GPU by default.

## # Requirements


Python 3.6 or later with the following `pip install -r requirements.txt` packages:

 - numpy
 - torch>=1.0
 - torchvision
 - tensorflow
 - pillow
 - tqdm
 - opencv



## Before testing

 - The input image format must be 'jpg' or 'png'. 
 
 - Store all images to be tested in the same directory</test_path>, and the default path is./test_data .
 
 - Create a directory </store_path> under the project directory to store the test results, and the default path is ./test_result .

## Testing

**Script:** test_my.py

**Optional args:**
 -  image_path  -path to test images
 - store_path  -path to test results
 - img_size  -size of each image dimension(no change recommended here)
 - n_cpu  -number of CPU threads to use during batch generation
 - pretrained_weights  -path to pretrained weights
 - model_def  -path to model definition file
 - batch_size  -size of each image batch
 
**Run:**
 
  `python test_my.py --image_path <test_path> --store_path <store_path> `
  You can got result images in output folder.
 ![enter image description here](http://a4.qpic.cn/psb?/V11FCSGJ2v3g58/or5Xx46iXRnmX0PbM0atazluSfz0gPVawwBChgCOpTA!/m/dFMBAAAAAAAAnull&bo=0AJAAgAAAAARB6A!&rf=photolist&t=5)

