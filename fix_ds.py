import json, numpy as np
from PIL import Image
import os
root = r'C:\\anomaly_detection\\data\\mvtec_ad_2'
data = json.load(open(root + '/meta.json'))
for cls, items in data['test'].items():
    has_mask = sum(1 for x in items if x['anomaly']==1 and x['mask_path']!='')
    print(f'{cls}: masks found = {has_mask}')