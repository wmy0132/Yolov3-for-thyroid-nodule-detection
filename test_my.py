from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from detect import detect_save
import os
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, default='pretrained_model/yolov3_ckpt.pth',
                        help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--image_path", type=str, default="test_data", help="path of test images")
    parser.add_argument("--store_path", type=str, default="test_result", help="path of test result")
    opt = parser.parse_args()
    print("Configurations:")
    print("#"*10)
    print(opt)
    print("#" * 10)

    # GPU config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data config
    # data_config = parse_data_config(opt.data_config)
    # test path config
    test_path = opt.image_path
    test_path_file = os.path.join(test_path, 'test.txt')
    # if os.path.exists(test_path_file):
    #     raise UserWarning('You may have tested this file before,please check again and delete the "test.txt" ')

    if os.path.exists(test_path):
        print('test path:', test_path)
        test_img_names = os.listdir(test_path)
        if len(test_img_names) == 0:
            raise FileNotFoundError('No image has been found in '+test_path+ ' ,detection failed.')
        num_of_images = 0
        test_list_file = open(test_path_file, 'w')
        for image_name in test_img_names:
            if image_name.split('.')[-1] == 'jpg' or image_name.split('.')[-1] == 'png':
                test_list_file.write(os.path.join(test_path, image_name) + '\n')
                num_of_images += 1
        test_list_file.close()
        if num_of_images == 0:
            raise FileNotFoundError('No image has been found in '+test_path+ ' ,detection failed.')
        else:
            print('Number of test images:', num_of_images)

    else:

        raise FileNotFoundError('No such file:'+test_path+' ,please check again!')


    # class names config
    class_names = ['Nudole']  # class.names
    # model config
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)
    if opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights))
        else:
            model.load_darknet_weights(opt.pretrained_weights)

    print("\n---- Evaluating Model(images) ----")
    # store path config
    store_path = opt.store_path
    detect_save(model, test_path_file, opt.img_size, opt.n_cpu, 1, store_path=store_path)






