import sys
sys.path.insert(0, "../../")
sys.path.insert(0, "../")

import numpy as np
from vs_model import VisualSearchModel as VisualSearchModel
from data_utils import get_data_paths, get_exp_info
import tensorflow as tf
import time

t_start = time.time()

physical_devices = tf.config.list_physical_devices('GPU')
for dev in physical_devices:
    tf.config.experimental.set_memory_growth(dev, True)

base_data_path = "../dataset/"

eccParam = {}
eccParam['rf_min'] = [2]*5
eccParam['stride'] = [2]*5
eccParam['ecc_slope'] = [0, 0, 3.5*0.02, 8*0.02, 16*0.02]
eccParam['deg2px'] = [round(30.0), round(30.0/2), round(30.0/4), round(30.0/8), round(30.0/16)]
eccParam['fovea_size'] = 4
eccParam['rf_quant'] = 1
eccParam['pool_type'] = 'avg'

model_desc = {
    'eccParam': eccParam,
    'ecc_depth': 5,
    'out_layer': [1, 1, 1],
    'comp_layer': 'diff',
    'vgg_model_path': "../../pretrained_model/vgg16_imagenet_filters.h5",
    'model_subname': "",
}

vsm = VisualSearchModel(model_desc)
print("Model loaded:", vsm.model_name)

exp_info = get_exp_info('categorization', base_data_path=base_data_path)
vsm.load_exp_info(exp_info, corner_bias=16*4*1)

stim_path, gt_path, tar_path = get_data_paths('categorization', 0, 0, base_data_path=base_data_path)
saccade = vsm.start_search(stim_path, tar_path, gt_path)

print("Saccade output shape:", saccade.shape)
print("Total time:", time.time() - t_start, "seconds")
print("SMOKE TEST PASSED")
