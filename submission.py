from __future__ import print_function

import numpy as np
import cv2
from data import image_cols, image_rows


### turn 0...1 logistic into a large-enough mask
def prep(img, threshold=0.75, minsize=0.015):
    img = img.astype('float32')
    img = cv2.threshold(img, threshold, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img, (image_cols, image_rows), interpolation=cv2.INTER_CUBIC)
    x = img.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < minsize*image_cols*image_rows:  # consider as empty
        img *= 0
    return img


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    #print (len(y))
    if len(y) == 0:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

def my_dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def calibrate():
    pred = np.load('imgs_mask_train.pred.fold1.npy')
    act  = np.load('imgs_mask_train.actual.fold1.npy')
    print(pred.shape)
    print(act.shape)
    score = 0
    total = act.shape[0]
    for thresh in [0.5]:
        for minsize in [0.001,0.01,0.02]:
            print("thresh: ", thresh)
            print("minsize: ", minsize)
            for i in range(total):
                pr = pred[i, 0]
                pr = prep(pr,thresh,minsize)
                ac  = act[i,0]
                #ac  = prep(ac,thresh,minsize)
#                print("mean pr: ", np.mean(pr))
#                print("mean ac: ", np.mean(ac))
                score += my_dice_coef(ac, pr)
            score /= total
            print("dice: ", score)

def submission():
    from data import load_test_data
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load('imgs_mask_test.npy')

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))
    count=0
    for i in range(total):
        if (len(rles[i])==0):
            count = count+1
    print ((1.0*count)/total)

    first_row = 'img,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    submission()
    #calibrate()
