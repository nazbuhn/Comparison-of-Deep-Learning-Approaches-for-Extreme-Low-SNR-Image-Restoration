import numpy as np

def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

def norm_minmse(y, x):
  x = np.squeeze(x)
  y = np.squeeze(y)
  y = normalize(y,0.1,99.9)
  x = x- x.mean()
  y = y-y.mean()
  scale=np.cov(x.flatten(), y.flatten())[0,1]/np.var(x.flatten())
  x = scale*x
  return y,x

# def save_img(filepath, img):
#     cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# def load_gray_img(filepath):
#     return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

# def save_gray_img(filepath, img):
#     cv2.imwrite(filepath, img)
