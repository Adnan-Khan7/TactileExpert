"""One definition of "which pixels are ink", shared by every script here.

This module exists because there were briefly two. `trivial_baselines.py` used a
plain dark-pixel fraction; `extract_morph_features.py` composited alpha, decided
polarity from the border, and stripped speckle. They agreed on 192 of 193 test
images and disagreed by 0.87 on the one white-on-black rendering
(`Vehicles_and_Flight_Systems/Ship/Tactile/5.jpg`), which the naive version
counts as 87% ink. That was enough to move the published `too_thick` baseline by
0.005 -- small, but it is the number an experiment is being measured against, so
it has to come from one place.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label


def otsu(gray: np.ndarray) -> int:
    """Otsu's threshold, guarding the degenerate tails.

    The textbook formula divides by w0*w1, which is zero at both ends of the
    histogram; unguarded, argmax returns 255 for every image and every mask
    downstream becomes the whole frame.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(float)
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m = np.cumsum(hist * np.arange(256))
    num = (m[-1] * w0 - m * w0[-1]) ** 2
    den = w0 * w1
    return int(np.argmax(np.where(den > 0, num / np.maximum(den, 1e-9), -np.inf)))


def load_gray(path, cap: int | None = 1024) -> np.ndarray:
    """Grayscale on a white ground, long side capped.

    Alpha compositing is not optional: a transparent PNG read as RGB gives a
    black background, which inverts the mask and turns every feature into its
    complement without raising anything.
    """
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    im = im.convert("L")
    if cap and max(im.size) > cap:
        s = cap / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return np.asarray(im)


def binarize(gray: np.ndarray, speckle_px: int = 0):
    """Ink mask plus the quality signals that say whether to trust it.

    Polarity is read off the border rather than assumed. The corpus is
    overwhelmingly dark-ink-on-white, but one test rendering is white-on-black,
    and without this it contributes a 97%-ink mask and a meaningless skeleton.
    """
    t = otsu(gray)
    dark = gray < t
    border = np.concatenate([dark[0], dark[-1], dark[:, 0], dark[:, -1]])
    inverted = bool(border.mean() > 0.5)
    mask = ~dark if inverted else dark

    # anti-aliasing and JPEG ringing live in a band around the threshold; a wide
    # band means any skeleton below is being drawn through mush
    band = max(8, int(0.15 * 255))
    ambiguity = float(((gray > t - band) & (gray < t + band)).mean())

    # the comparator published in README 4h is the fraction BEFORE cleanup
    ink_raw = float(mask.mean())

    speckle = 0
    if speckle_px > 0 and mask.any():
        lab, n = cc_label(mask)
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = sizes >= speckle_px
            speckle = int((~keep[1:]).sum())
            mask = keep[lab]
    return mask, t, inverted, ambiguity, speckle, ink_raw


def ink_fraction(path, cap: int | None = 1024) -> float:
    """The README 4h baseline: polarity-aware, pre-cleanup ink fraction."""
    return binarize(load_gray(path, cap))[5]
