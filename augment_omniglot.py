import os
import sys
import skimage as sk
from skimage import io
# print(os.getcwd())
import glob
import numpy as np
from img_aug import random_transform, remove_similar
import matplotlib.pyplot as plt

MAX_FILES = 60
GEN_IMGS = 3 * MAX_FILES
N_TRANS = 3

centered = bool(1)
in_path = '../omniglot/python/images_background/'
out_path = '../omniglot/augmented/'
os.makedirs(out_path, exist_ok=True)
alphabets = sorted(glob.glob(os.path.join(in_path, '*')))
# print(folders)
# print(classes)
size = 32
# size = 64

for alpha in alphabets:
    alpha_name = os.path.basename(alpha)
    print(alpha_name)
    out_alpha = os.path.join(out_path, alpha_name)
    os.makedirs(out_alpha, exist_ok=True)

    characters = sorted(glob.glob(os.path.join(alpha, '*')))
    for char in characters:
        char_name = os.path.basename(char)
        print(char_name)
        out_char = os.path.join(out_alpha, char_name)
        os.makedirs(out_char, exist_ok=True)

        filenames = sorted(glob.glob(os.path.join(char, '*.png')))
        for samp_idx, fname in enumerate(filenames):
            img = sk.color.rgb2gray(sk.io.imread(fname)) == 0
            small = sk.transform.resize(img, (size, size), anti_aliasing_sigma=0.1)

            out_fname = os.path.join(out_char, "{:06d}.png")

            aug = [random_transform(small, N_TRANS) for _ in range(GEN_IMGS)]
            aug = remove_similar(aug)
            # print(len(aug))

            # sk.io.imshow(aug[0], cmap='gray')

            for aug_idx, img in enumerate(aug[:MAX_FILES]):
                name = out_fname.format(MAX_FILES * samp_idx + aug_idx)
                # print(name)
                sk.io.imsave(name, (img * 255.).astype('uint8') )
                # sys.exit(0)

            # break
