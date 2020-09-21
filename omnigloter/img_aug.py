import skimage as sk
from skimage import transform
from skimage import util
from skimage import filters
import numpy as np
from numpy import ndarray
from scipy.signal import correlate2d
from copy import copy


def vflip(img: ndarray):
    return img[::-1, :]


def hflip(img: ndarray):
    return img[:, ::-1]


def rotate(img: ndarray, min_deg: float = -10., max_deg: float = 10.):
    whr = np.where(img != 0)
    r, c = np.mean(whr[0]), np.mean(whr[1])
    angle = np.random.uniform(min_deg, max_deg)
    return sk.transform.rotate(img, angle, center=(r, c))


def sp_noise(img: ndarray):
    return sk.util.random_noise(img, mode='s&p')


def gauss_noise(img: ndarray):
    return sk.util.random_noise(img, mode='localvar')


def gauss_filter(img: ndarray, min_sigma: float = 0.1, max_sigma: float = 3.):
    sigma = np.random.uniform(min_sigma, max_sigma)
    return sk.filters.gaussian(img, sigma)


def shift(img: ndarray, min_dx: float = 0.5, max_dx: float = 2.):
    dx = np.random.uniform(min_dx, max_dx, size=2)
    tform = sk.transform.AffineTransform(translation=dx)
    return sk.transform.warp(img, tform.inverse)


def scale(img: ndarray, min_s: float = 0.75, max_s: float = 1.1):
    h, w = img.shape
    h = (h + 1.) * 0.5
    w = (w + 1.) * 0.5
    s = np.random.uniform(min_s, max_s, size=2)
    mtx = np.array([[s[0], 0, (-s[0] * w) + w],
                    [0, s[1], (-s[1] * h) + h],
                    [0, 0, 1]])
    tform = sk.transform.AffineTransform(mtx)
    return sk.transform.warp(img, tform.inverse)


def shear(img: ndarray, min_s: float = 0.01, max_s: float = 0.1):
    h, w = img.shape
    r = (h + 1.) * 0.5
    c = (w + 1.) * 0.5
    ctr = sk.transform.AffineTransform(translation=(-r, -c))
    rectr = sk.transform.AffineTransform(translation=(r, c))
    s = np.random.uniform(min_s, max_s, size=1)
    tform = sk.transform.AffineTransform(shear=s)
    tform = ctr + tform + rectr
    return sk.transform.warp(img, tform.inverse)


TRANSFORMS = {
    #     'vflip': (vflip, None),
    #     'hflip': (hflip, None),
    'rotate': (rotate, (-20., 20.)),
    #     's&p': (sp_noise, None),
    #     'noise': (gauss_noise, None),
    #     'smooth': (gauss_filter, (0.25, 1.)),
    'shift': (shift, (-3., 3.)),
    'scale': (scale, (0.75, 1.1)),
    'shear': (shear, (0.01, 0.5))
}


def random_transform(img: ndarray, n: int = 1, xform: dict = TRANSFORMS):
    keys = sorted(xform.keys())
    replace = (n > len(xform))
    ts = np.random.choice(np.arange(len(xform)), size=n, replace=replace)
    _img = img.copy()
    for ki in ts:
        #         print(ki, keys[ki])
        k = keys[ki]
        f, p = xform[k]
        _img[:] = f(_img, p[0], p[1]) if not p is None else f(_img)

    return _img


def remove_similar(augmented: list, similar_thresh: float = 0.8):
    td = []
    for i0, img0 in enumerate(augmented):
        for i1, img1 in enumerate(augmented):
            if i1 < i0:
                continue
            m = np.sqrt((img0 ** 2).sum() * (img1 ** 2).sum())
            c = correlate2d(img0, img1, mode='valid') * (1. / m)
            if i0 != i1 and c >= similar_thresh:
                td.append(i1)

    aug_clean = copy(augmented)
    for d in sorted(np.unique(td))[::-1]:
        del aug_clean[d]

    return aug_clean
