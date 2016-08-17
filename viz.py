from __future__ import print_function

import numpy as np
import train
import cv2
from data import image_cols, image_rows
from submission import prep
from submission import run_length_enc

def vis():
    from data import load_test_data
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load('imgs_mask_test.npy')
    total = imgs_test.shape[0]
    for i in range(total):
        img = imgs_test[i, 0]
        pimg = prep(img)
        rle = run_length_enc(pimg)
        img = pimg.astype(np.float32)
        if (len(rle)==0):
            img=img*0
        else:
            img=img*256
        cv2.imwrite(str(i) +".png", img)

vis()
