#!/usr/bin/env python
# coding: utf-8

# In[1]:

# VERSION 12/11/2024
# returns empty mask for lens if no contours found
# VERSIOM 10/11/2024
# added fluorescense channel for 20X

# VERSION 19/09/24
# added stitching with exposure compensation
# added line to fish_prep to more accurately capture dorcel fin
# replaced gfilt with gfilt_cv in some of the cases

import functools
import itertools
import os
import shutil
import subprocess
import tkinter as tk
import xml.etree.ElementTree as ET
from functools import reduce
from inspect import getsource
from itertools import count
from os.path import join as pjoin
from pathlib import Path
from shutil import copy as pcopy
from time import sleep, time
from tkinter import filedialog
from tkinter.ttk import Progressbar

import customtkinter as ctk
import cv2 as cv2
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import xmltodict as x2d
from aicspylibczi import CziFile as czi
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from numpy import cos, exp
from numpy import log as ln
from numpy import log2
from numpy import log10 as log
from numpy import sin, sqrt, tan
from pandas import DataFrame as df
from PIL import Image, ImageDraw, ImageFile, ImageFilter, ImageOps, ImageTk
from PIL.ImageTk import PhotoImage
from pyperclip import copy as _copy
from pyperclip import paste
from scipy.ndimage import gaussian_filter as gfilt
from scipy.ndimage import grey_dilation as g_dil
from scipy.stats import rankdata


# In[2]:


def precede(path, n=1):
    """returns the path after discarding the last n items (climbs up the tree n steps)."""
    if n <= 0:
        raise Exception("non-positive precede")
    return pjoin(*Path(path).parts[:-n])


def istrue(x, v=0):
    """returns truthness of a value, or of a list or array of values. not sure how it will work on multidimensional (>2)"""
    x = np.array(x)
    shape = x.shape
    if len(shape) == 0:
        if x:
            if np.isnan(x):
                return False
            return True
        else:
            return False
    else:
        return np.array([istrue(a) for a in x.flatten()]).reshape(shape)


def qnorm(x):
    """squeezes values between 0 and 1, ignores false numbers (None, nan). doesn't ignore strings."""
    x = np.array(x)
    tmp = x - x[np.logical_or(x == 0, istrue(x))].min()
    try:
        tmp = tmp / tmp[istrue(tmp)].max()
    except:
        tmp = tmp / tmp.max()
    return tmp


def piximg(x, f):
    """pixelate downscale img x by factor f"""
    return np.array(Image.fromarray(x).resize((x.shape[0] // f, x.shape[1] // f)))


def list_concat(lst):
    """concatenates lists of lists to one list"""
    nlst = []
    lst = list(lst)
    for x in lst:
        for y in x:
            nlst.append(y)
    return nlst


def filtl(*args, **kwargs):
    return list(filter(*args, **kwargs))


def gstretch(x, s=255):
    """stretch to range"""
    return (qnorm(x) * s).astype(np.uint8)


def flat_dir(path, dirs=0):
    return list_concat([[pjoin(x[0], y) for y in x[2 - dirs]] for x in os.walk(path)])


def phead(path):
    return Path(path).name


def headtail(head="", tail=""):
    lh = len(head)
    lt = len(tail)
    return lambda x: x[:lh] == head and x[len(x) - lt :] == tail


def LS(path, dirs=0, dpath=0):
    """returns np.array of items in path"""
    func = [lambda x: x.name, lambda x: x.path][dpath]
    if dirs:
        return np.array([func(x) for x in os.scandir(path) if x.is_dir()])
    return np.array([func(x) for x in os.scandir(path) if not x.is_dir()])


def project_czi(path, method=np.max, channel=1, retstack=0, full_stack=0):
    """returns projection of czi in path, using method for the projection, in channel
    param retstack - flag, enter 1 to get the whole stack
    param fullstack - flag, enter 1 to get all the stacks within the file"""
    tfile = czi(path)
    stack = tfile.read_image(B=0, H=0, T=0)[0]
    return stack[0, 0]
    while stack.shape[0] == 1:
        stack = stack[0]
    if retstack:
        if full_stack:
            return stack
        else:
            return stack[channel]
    return method(stack[channel], 0)


def save_df(path, d):
    """accepts a path and a pandas DataFrame
    saves as a np.array where the first row is the dataframe column names
    load using load_df"""
    dat = np.array(d)
    key = np.array(d.keys())
    arr = np.concatenate(([key], dat), 0)
    np.save(path, arr, allow_pickle=True)
    return path


def load_df(path):
    """loads a npy file in location path, creates a dataframe where the column names are the values in the first row"""
    f = np.load(path, allow_pickle=True)
    key, dat = f[0], f[1:]
    return df(dat, columns=key)


def qfilt(x, vmin=0, vmax=1):
    """Nones out values outside range vmin-vmax"""
    x = np.array(x) / 1
    filt = np.logical_or(x < vmin, x > vmax)
    x[filt] = None
    return x


def nonone(x):
    """zeros out all falsey values, including 0, False, none, nan..."""
    x[np.logical_not(istrue(x, 1))] = 0
    return x


def erode(x, s, n):
    """erode x with kernel size s*s with n iterations"""
    kern = np.ones((s, s))
    return cv2.erode(x, kernel=kern, iterations=n)


def g2bgr(x):
    """gray (1 channel) to bgr (3 channels)"""
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)


def bgr2g(x):
    """bgr (3 channels) to gray (1 channel)"""
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

def gfilt_cv(x, s):
    return cv2.GaussianBlur(x, (0, 0), s)
    
def dog(x, a, b):
    return gfilt_cv(x, a) - gfilt_cv(x, b)


def czi_meta(x, path=0):
    if path:
        x = czi(x)
    return x2d.parse(ET.tostring(x.meta[0]))["Metadata"]


def is_notebook() -> bool:
    """checks if the script is beaing run in a notebook"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


class ProgressBar(Progressbar):
    """class to make progressbar that updates with every step"""
    def step(self, amount=1):
        """Increments the value option by amount.

        amount defaults to 1.0 if omitted."""
        self.tk.call(self._w, "step", amount)
        print(self.cget("value"))
        self.master.update_idletasks()


def set_df_element(d, col, row, val):
    """set value val in dataframe d in column col in row row. can use indices or keys for col, row only index.
    returns the whole dataframe with the new element in place"""
    if type(col) is int:
        col = d.keys()[col]
    ls = list(d[col])
    ls[row] = val
    d[col] = ls
    return d


def lens_prep(r, show=0, retshow=0, todraw=0):
    """given X20 zoom image of an eye, attempts to find iris and crystal containing portion of the eye
    r is the image as grayscale np.ndarray
    show is flag, retshow returns displayable image, todraw is optional image to draw the contours upon"""
    r1 = g_dil(gstretch((r > 6) * 1), 4)
    r2 = erode(r1, 2, 4)
    r3 = 255 - erode(255 - r2, 5, 3)
    # r3 = gfilt(r3 - gfilt(r3, 1), 1)

    cnt, hiers = conts(r3, retcont=1)
    try:
        ri = np.argmax([cv2.contourArea(cnt[i]) for i in range(len(cnt))])
    except:
        return r1 * 0
    ccc = cnt[ri]
    inners = []
    inners.extend([x for i, x in enumerate(cnt) if hiers[0, i, 3] == ri])
    try:
        inner = inners[
            np.argmax([cv2.contourArea(inners[i]) for i in range(len(inners))])
        ]
    except:
        inner = 0
    r4 = gfilt(r3, 6)
    cnt, hiers = conts(r4, retcont=1)
    ri = np.argmax([cv2.contourArea(cnt[i]) for i in range(len(cnt))])
    ccc = cnt[ri]
    k = cv2.drawContours(
        r3 * 0, contours=[ccc], contourIdx=0, color=(255, 255, 255), thickness=-1
    )
    if not type(inner) is int:
        mask = cv2.drawContours(
            k.copy(), contours=[inner], contourIdx=-1, color=(0, 0, 0), thickness=-1
        )
        cont_disp = [ccc, inner]
    else:
        mask = k
        cont_disp = [ccc]
    if show:
        if type(todraw) is int:
            todraw = g2bgr(((r > 10) * 255).astype(np.uint8))
        if len(todraw.shape) < 3:
            todraw = g2bgr(todraw)
        show = cv2.drawContours(
            todraw, contours=cont_disp, contourIdx=-1, color=(0, 255, 0), thickness=2
        )
        if retshow:
            return show
        plt.imshow(show)
        plt.show()
    return mask


def conts(img, lb=127, ub=255, retcont=0, to_draw=0, width=3):
    """finds contours with option to either return the contours with hierarchy or return image to_draw with contours in the image
    params lb ub : lower and upper bounds
    param retcont : boolean, whether to return the contours in tuple of lists, or return image with contours drawn on.
    param to_draw : image to draw the contours on, optional.
    param width : width of contours to draw. negative value fills the contour."""
    t = img.copy()
    ret, thresh = cv2.threshold(t, lb, ub, 0)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if type(to_draw) is int:
        to_draw = t
    cv2.drawContours(to_draw, contours, -1, (0, ub, 0), thickness=width)
    return [to_draw, (contours, hierarchy)][retcont]


def fish_prep(r):
    """attempts find fish contours"""
    r = r * (dog(r, 10, 11) > 100) # remove transparent areas by removing thin areas
    r = qfilt(r, 1, 200)  # remove bright colours
    r = nonone(g_dil(r, 0) - g_dil(r, 10))  # find expanded area where there is a fish
    r = dog(r, 1, 7)  # find sharp edges
    r = cv2.Canny(
        gfilt_cv(cv2.Canny(gfilt((qnorm(r) * 255).astype(np.uint8), 1), 10, 80), 5), 20, 20
    )  # find high complexity areas
    r = gfilt_cv(g_dil(r, 8), 3)  # expand the high complexity areas
    return r


def eye_prep(r):
    """attempts to find eye contours"""
    r = -g_dil(-r, 10)  # expand dark areas
    r = conts(gfilt(r, 3), 50, 200)  # find contours after blurring the image
    r = g_dil(r, 10)  # expand bright areas
    r = ((r < 30) * 255).astype(np.uint8)  # make mask of the dark areas
    r = erode(r, 15, 3)  # remove small area masks
    r = g_dil(r, 30)  # expand remaining mask
    return r


def find_length(r, show=0, prep="fish", retshow=0):
    """find length of object in image r, using method in prep
    param show : boolean if to show image or return the length
    param retshow : boolean, if to show the displayable image or return it
    param fish : str flag, 'fish' for fish (X2) and 'eye' for eyes (X10)"""
    preps = ["fish", "eye"]
    ind = preps.index(prep)
    rp = [fish_prep, eye_prep][ind](r)
    try:
        return maxr_cont(rp, r, show=show, retshow=retshow)
    except Exception as e:
        print(e)
        return None


def maxr_cont(rp, r, show=0, retshow=0):
    """finds contours in rp, chooses one with largest enclosing circle radius
    param rp : image to find the contours int
    param r : image to draw upon
    param show : boolean, whether to show the image or return diameter
    param retshow : boolean, whether to show the image with circle or return it
    get contours"""
    cnt, hier = conts(rp, retcont=1)
    # find larget contour
    try:
        ri = np.argmax([cv2.minEnclosingCircle(cnt[i])[1] for i in range(len(cnt))])
    except:
        if show:
            if retshow != 0:
                retshow = r
                return r
        return None
    (x, y), radius = cv2.minEnclosingCircle(cnt[ri])
    center = (int(x), int(y))
    radius = int(radius)
    if len(r.shape) > 2:
        r = bgr2g(r)
    if show:
        if retshow != 0:
            retshow = cv2.circle(g2bgr(r).copy(), center, radius, (0, 255, 0), 2)
            return retshow
        fig, ax = plt.subplots(1, 2, figsize=(10, 20))
        ax[0].imshow(conts(rp, to_draw=g2bgr(r).copy()))
        ax[1].imshow(cv2.circle(g2bgr(r).copy(), center, radius, (0, 255, 0), 2))
        plt.show()
    return radius * 2


def czi_px_size(x, path=0):
    """return pixel size in a czi file.
    x is a czi object, but optionally can use a path directly,
    in which case path should be True."""
    return float(czi_meta(x, path)["Scaling"]["Items"]["Distance"][0]["Value"])


def ls2npims(x):
    """accepts a path and returns the image in that path as a 2-D np.ndarray"""
    if type(x) is str:
        if Path(x).exists():
            tmp = np.array(Image.open(x))
            if len(tmp.shape) > 2:
                tmp = bgr2g(tmp)
            return tmp
    return None


def ctkbutton(master, *args, **kwargs):
    """utility to make consistent ctk buttons"""
    return ctk.CTkButton(
        master=master, height=28, width=len(kwargs["text"]) * 5, *args, **kwargs
    )


def pack_widgets(ls, side="right", padx=2, pady=0):
    """accepts a list of widget objects and packs them all sequentially"""
    [x.pack(side=side, padx=padx, pady=pady) for x in ls]
    return


def pack_widgs(x):
    """consistently pack widgets side by side within a frame"""
    pack_widgets(x, padx=2, pady=0, side="left")


def pack_frames(x):
    """consistently pack frames one on top of the other"""
    pack_widgets(x, padx=0, pady=5, side="top")


def stitch(imgs, vertical = 1):
    img1, img2 = imgs
    rotate = lambda x: x if vertical else x.T
    if len(img1.shape) == 2:
        img1 = g2bgr(rotate(img1))
    if len(img2.shape) == 2:
        img2 = g2bgr(rotate(img2))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Get matched keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find affine transformation
    affine_matrix, inliers = cv2.estimateAffine2D(dst_pts, src_pts)

    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the new dimensions after transformation
    corners = np.array([[0, 0], [0, h2-1], [w2-1, h2-1], [w2-1, 0]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners, affine_matrix)
    min_x = min(0, np.min(transformed_corners[:, 0, 0]))
    max_x = max(w1, np.max(transformed_corners[:, 0, 0]))
    min_y = min(0, np.min(transformed_corners[:, 0, 1]))
    max_y = max(h1, np.max(transformed_corners[:, 0, 1]))

    # Create the stitched image
    stitched_width = int(max_x - min_x)
    stitched_height = int(max_y - min_y)
    result = np.zeros((stitched_height, stitched_width, 3), dtype=np.uint8)

    # Adjust the affine matrix for the new image size
    affine_matrix[0, 2] -= min_x
    affine_matrix[1, 2] -= min_y

    # Apply affine transformation to img2
    img2_transformed = cv2.warpAffine(img2, affine_matrix, (stitched_width, stitched_height))

    # Copy img1 to the result
    result[-int(min_y):h1-int(min_y), -int(min_x):w1-int(min_x)] = img1

    # Create a mask for smooth blending
    mask = np.zeros((stitched_height, stitched_width), dtype=np.float32)
    cv2.warpAffine(np.ones((h2, w2), dtype=np.float32), affine_matrix, (stitched_width, stitched_height), dst=mask)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=20, sigmaY=20)

    # Blend the images
    result = result * (1 - mask[:, :, np.newaxis]) + img2_transformed * mask[:, :, np.newaxis]

    return (0, bgr2g(result.astype(np.uint8)))


# In[479]:


class QWidget:
    # a class that creates instances of the EyeQWidget

    px_size = 1.2478515625e-05  # confocal pixel size, with enlargement 1 (divide by enlragement to obtain the pixel size of an image)
    pxsize = {"X2": px_size / 2.5, "X10": px_size / 10, "X20": px_size / 20}

    def __init__(self):
        # create widget main window
        ctk.set_appearance_mode("dark")
        self.root = ctk.CTk()
        self.root.title("Image Manipulation Widget")
        self.root.protocol(
            "WM_DELETE_WINDOW", lambda: self.close_q(self.root, main=1)
        )  # bind window colse to a prompt if to close the window

        # set naming template
        self.naming_convention = "group_age_zoom_part_id_comment_collector"
        self.columns = ["path"] + self.naming_convention.split("_")
        self.instab_columns = "path_group_id_age_zoom_channel".split("_")
        self.stab_zooms = ["X2", "X2b", "X10", "X20", "X40"]  # list of zooms to accept
        self.view_zooms = [
            "X2_0",
            "X2b_0",
            "X10_0",
            "X20_0",
            "X20_1",
        ]  # list of zooms to show in image showing window

        # constants
        self.x20_thresh = 6
        self.x20_thresh_fluor = 6

        # buttons
        self.buttons_frame = ctk.CTkFrame(self.root)
        self.tab_button = ctkbutton(
            self.buttons_frame,
            text="make image/stack-per-row table",
            command=self.mktab,
        )
        self.tab2singles_button = ctkbutton(
            self.buttons_frame,
            text="make single images directory",
            command=self.tab2singles,
        )
        self.stab_button = ctkbutton(
            self.buttons_frame, text="make sample per row table", command=self.mkstab
        )
        self.make_auto_button = ctkbutton(
            self.buttons_frame, text="automate masks", command=self.automasks
        )
        self.show_stab_button = ctkbutton(
            self.buttons_frame,
            text="show sample per row images",
            command=self.show_stab,
        )
        self.load_data_button = ctkbutton(
            self.buttons_frame, text="load data", command=self.load_data
        )
        self.show_thresh_button = ctkbutton(
            self.buttons_frame, text="show thresholded", command=self.show_thresh
        )
        self.get_results_button = ctkbutton(
            self.buttons_frame, text="get results", command=self.mkresults
        )
        self.eye20_thresh_button = ctkbutton(
            self.buttons_frame,
            text="20X paint threshold",
            command=self.def_eye20_thresh,
        )
        pack_widgs(
            [
                self.tab_button,
                self.tab2singles_button,
                self.stab_button,
                self.show_stab_button,
                self.make_auto_button,
                self.load_data_button,
                self.show_thresh_button,
                self.get_results_button,
                self.eye20_thresh_button,
            ]
        )
        pack_frames([self.buttons_frame])

        # label for information messages, such as progress, warnings etc.
        self.infolabel = ctk.CTkLabel(self.root, text=self.naming_convention)
        pack_frames([self.infolabel])

    def def_eye20_thresh(self):
        tempthresh = tk.simpledialog.askfloat(
            "threshold eyes",
            "enter threshold value from 0 to 255 (0 is lowest possible value).",
        )
        self.x20_thresh = tempthresh

    def close_q(self, root, main=0):
        # function to prompt destroying a window
        # param root : Tk or Toplevel widget to destroy
        # param main : flag if to destroy the whole widget (main window) and quit the script or just destroy a specific window
        win = tk.Toplevel(master=root)
        close_button = ctkbutton(
            win, text="Close", command=lambda: self.close(self.root, main=main)
        )
        close_button.pack(side="right", padx=2, pady=5)
        cancel_button = ctkbutton(win, text="Cancel", command=win.destroy)
        cancel_button.pack(side="left", padx=2)

    def close(self, root, main=0):
        # function that destroys the window given in argument root
        # param main : flag if to quit python program (when closing main window) or just destroy specific window
        root.quit()
        root.destroy()
        if (
            main and not is_notebook()
        ):  # if python runs not in a notebook, kill the kernel
            quit()
            sexit()

    def start(self):
        # reinitiate the widget and run it
        self.__init__()
        self.root.mainloop()

    def mktab(self):
        # accepts a directory and makes a 'tab.csv' (table with one row per image with properties of the image) file
        # using the files in the directory, according to the template in self.columns
        self.tabdir = tk.filedialog.askdirectory()
        self.tab = df(
            [[x] + Path(x).stem.split("_") for x in LS(self.tabdir)],
            columns=self.columns,
        )
        # self.tab = self.tab[self.tab.iloc[:, 0] != 'tab.csv']
        self.tab.to_csv(pjoin(self.tabdir, "tab.csv"))

    def tab2singles(self):
        # accepts a 'tab.csv' file and saves projections and single images, named according to the tab.csv
        # inside 'singles' directory, within the directory where tab.csv is
        self.tabdir = Path(tk.filedialog.askopenfilename()).parent
        self.tab = pd.read_csv(self.tabdir / "tab.csv")
        # make singles directory
        single_dir = self.tabdir / "singles"
        if not single_dir.exists():
            single_dir.mkdir()
        # make projections or just copies of the files specified in tab.csv
        for i in self.tab.iterrows():
            k = i[1]
            zzoom = f'{k.zoom}{k.part * (k.part == "b")}'
            name = f"{k.group}_{k.id}_{k.age}_{zzoom}"
            #             name = '_'.join([str(x) for x in k[2:]])
            try:
                projections = project_czi(
                    self.tabdir / k.path, retstack=1, full_stack=1
                ).max(1)
                proj = [
                    Image.fromarray(x).save(
                        self.tabdir / "singles" / (name + "_" + str(j) + ".png")
                    )
                    for j, x in enumerate(projections)
                ]
            except Exception as e:
                print(f"{e} - {k}")
        # make 'stab.csv' file (table with all the images paths from each specimen in one row)
        self.mkstab(single_dir)

    def mkstab(self, path=0):
        # accepts a path and makes a 'stab.csv' file (table with all the images paths from each specimen in one row)
        # can also run with an input path, used for making new stab files within the program
        if path:
            self.stabdir = path
        else:
            self.stabdir = Path(tk.filedialog.askdirectory())
        # make a tab.csv file from all the png files anywhere in the directory and subdirectory
        self.tab = df(
            [
                [x] + Path(x).stem.split("_")
                for x in filtl(headtail("", ".png"), flat_dir(self.stabdir))
            ],
            columns=self.instab_columns,
        )
        self.tab.to_csv(self.stabdir / "tab.csv")
        # columns including two columns per zoom, so there is space for different channels
        self.stab_cols = list(
            np.array([f"{x}_0 {x}_1".split(" ") for x in self.stab_zooms]).flatten()
        ) + ["age", "group", "id"]
        # if there is a 'masks' directory in 'singles', make appropriate columns for the files in it too
        if "masks" in LS(self.stabdir, dirs=1):
            self.stab_cols += [x + "_m" for x in QWidget.DISPLABELS]
        # make empty df with columns in self.stab_cols and fill it
        self.stab = df(columns=self.stab_cols)
        for i in self.tab.iterrows():
            k = i[1]
            label = f"{k.group}_{k.age}_{k.id}"
            zoomchannel = f"{k.zoom}_{k.channel}"
            if label not in self.stab.index:
                self.stab.loc[label] = ""
                self.stab.loc[label, "age"] = k.age
                self.stab.loc[label, "group"] = k.group
                self.stab.loc[label, "id"] = k.id
            self.stab.loc[label, zoomchannel] = k.path
        self.stab.to_csv(self.stabdir / "stab.csv")

    def show_tab(self):
        # function to show the images in 'tab.csv'
        # realized there is no need for this function but leaving it here in case someone decides it's useful
        pass

    def mkstabis(self):
        # function to create a stabi object from a 'stab.csv' file
        # the object contains the actual images and any other data or information
        cols = self.view_zooms  # columns to be used in analysis and viewing
        # add masks columns, if there are
        if "masks" in LS(self.stabdir, dirs=1):
            cols += [x + "_m" for x in QWidget.DISPLABELS if x not in 'lensdisplayf']
        # turn paths to np.array images
        self.stabi = self.stab[cols].applymap(self.ls2npims)
        # restore label, age, id and group after corrupting them in previous line
        self.stabi["label"] = self.stab[self.stab.keys()[0]]
        self.stabi["age"] = self.stab.age
        self.stabi["id"] = self.stab.id
        self.stabi["group"] = self.stab.group

    def mkstabI(self):
        # makes a stabI object, that contains all the images of stabi object, but in PhotoImage format, for tk
        self.stabI = self.stabi.applymap(self.np2photo)
        print(self.stabI.keys())
        self.stabI["label"] = self.stabi["label"]
        self.stabI["age"] = self.stabi.age
        self.stabI["id"] = self.stabi.id
        self.stabI["group"] = self.stabi.group
        self.stabI = self.stabI.sort_values(by=["age", "group", "id"])
        self.stabI = self.stabI[
            [
                x
                for x in ["label"]
                + self.view_zooms
                + [
                    "X2s_m",
                    "fishdisplay_m",
                    "eyedisplay_m",
                    "lensdisplay_m",
                    "X2s",
                    "fishdisplay",
                    "eyedisplay",
                    "lensdisplay",
                    "age",
                    "group",
                    "id",
                ]
                + (['lensdisplayf'] if "X20_1" in self.view_zooms else [])
                if x in self.stabI.keys()
            ]
        ]

    DISPLABELS = [
        "lensdisplay",
        "lensdisplayf",
        "fishdisplay",
        "eyedisplay",
        "X2s",
    ] # column labels to add to the tabels once masks exist

    def automasks(self):
        # automatically generate masks and measurements accordig to a stab.csv file
        # will attempt to keep a progressbar running, but since it is a heavy function it might not update, but the widget will still work.
        self.stabdir = Path(tk.filedialog.askopenfilename()).parent
        self.stab = pd.read_csv(self.stabdir / "stab.csv")
        # make progress bar
        prog = ProgressBar(self.root, maximum=8)
        pack_frames([prog])
        prog.step()
        sleep(2)
        # make stabi object
        self.mkstabis()
        # stitch the X2 images
        self.mkstitches()

        # make X20 masks
        self.stabi["lensmasks"] = [
            lens_prep(x) if type(x) is np.ndarray else None for x in self.stabi.X20_0
        ]
        prog.step()
        # make displayable X20 masking results
        self.stabi["lensdisplay"] = [
            lens_prep(x, show=1, retshow=1).astype(np.uint8)
            if type(x) is np.ndarray
            else None
            for x in self.stabi.X20_0
        ]
        if 'X20_1' in self.view_zooms:
            self.stabi['lensdisplayf'] = self.stabi['X20_1'] * self.stabi['lensmasks'].map(lambda x: x.astype(bool))
        prog.step()

        # measure fish length on X2 images
        self.stabi["fishlength"] = [
            find_length(x, prep="fish") if type(x) is np.ndarray else None
            for x in self.stabi.X2s
        ]
        prog.step()
        # make displayable X2 fish circling
        self.stabi["fishdisplay"] = [
            find_length(x, prep="fish", show=1, retshow=1)
            if type(x) is np.ndarray
            else None
            for x in self.stabi.X2s
        ]
        prog.step()

        # measure eye max diameter
        self.stabi["eyelength"] = [
            find_length(x, prep="eye") if type(x) is np.ndarray else None
            for x in self.stabi.X10_0
        ]
        prog.step()
        # make displayable X10 eye circling
        self.stabi["eyedisplay"] = [
            find_length(x, prep="eye", show=1, retshow=1)
            if type(x) is np.ndarray
            else None
            for x in self.stabi.X10_0
        ]
        prog.step()

        # make a directory with the displayable images
        maskdir = self.stabdir / "masks"
        if maskdir.exists():
            shutil.rmtree(maskdir)
        maskdir.mkdir()

        # turn np.array images into image files in mask directory
        for i, label in enumerate(self.stabi.label):
            for disp in QWidget.DISPLABELS:
                aimage = self.stabi[disp][i]
                group, iid, age = label.split("_")
                if type(aimage) is np.ndarray:
                    name = maskdir / f"{group}_{age}_{iid}_{disp}_m.png"
                    Image.fromarray(aimage).save(name)
        prog.destroy()
        self.mkstab(self.stabdir)

        # save the stabi object as stabi.npy
        self.save_data()

    def mkstitches(self):
        # ask user if the stitching direction
        vertical = tk.simpledialog.askinteger(
            "stitching direction", "to stitch horizontally, type 0. vertical, type 1."
        )
        stitches = []
        # in each row, run stitch function on X2 parts a and b
        for i, x in enumerate(self.stabi.iterrows()):
            row = x[1]
            imglist = [
                x
                for x in [row[self.view_zooms[0]], row[self.view_zooms[1]]]
                if type(x) is np.ndarray
            ]
            if len(imglist) > 1:
                try:
                    res = stitch(imglist, vertical=vertical)
                    if res[0]:  # cv raised a stitch error
                        raise Exception(f"Not Enough to stitch {i}")
                    else:
                        stitches.append(res[1])
                except Exception as e:  # unsuccessful stitching
                    stitches.append(imglist[0])
                    print(e)
            elif len(imglist) == 1:
                stitches.append(imglist[0])
            else:
                stitches.append(None)
        self.stabi["X2s"] = stitches
        return

    def show_stab(self, fromstab=1):
        # show images from paths in a stabfile
        # param fromstab : boolean, if to use images from paths in a stab.csv file, or if to use the current stabi object
        self.MAN_MASKS = {
            "X2s": self.manstitch,
            "fishdisplay": self.paint,
            "eyedisplay": self.paint,
            "lensdisplay": self.bipaint,
            "lensdisplayf": self.empty_func
        }
        # fetch stab.csv file
        if fromstab:
            self.stabdir = Path(tk.filedialog.askopenfilename()).parent
            self.stab = pd.read_csv(self.stabdir / "stab.csv")
            self.mkstabis()
        # make viewing window
        self.nwin = tk.Toplevel()
        self.iframe = ctk.CTkScrollableFrame(self.nwin, width=2000, height=2000)
        pack_frames([self.iframe])
        self.rows = []
        rows = self.rows
        # update stabI object by stabi object
        self.mkstabI()
        print(self.stabI.keys())
        for i, x in enumerate(self.stabI.iterrows()):
            # make frame for each sample and label for each image
            row = x[1]
            rowframe = ctk.CTkFrame(self.iframe)
            rowlabs = [ctk.CTkLabel(rowframe, text=row.label)]
            for j, k in enumerate(row.keys()):
                if k in ["id", "age", "group", "label"]:
                    continue
                rowlabs.append(
                    ctk.CTkLabel(rowframe, image=row[k], text="", height=120, width=120)
                )
                rowlabs[-1].bind("<Button-1>", self.man_masks(i, j, row, k))
            pack_frames([rowframe])
            self.rows = rowlabs
            pack_widgs([x for x in rowlabs if str(type(x)) != "<class 'function'>"])
            rows.append((rowframe, rowlabs))
        self.rows = rows

    def empty_func(self, *args, **kwargs):
        return
    
    def man_masks(self, i, j, row, k):
        # function to return the right function to call an edit on an automatically generated image.
        # uses an inner function in order to make a new function each call (fails when using lambda function)
        # params i, j, row, k : rownum in stabI, colnum in stabI, row from stabI, colname in stabI respectively
        k = k.split("_m")[0]

        def inner(x):
            print(k)
            if k not in self.MAN_MASKS:
                return
            print(f"function {k}")
            return self.MAN_MASKS[k](i, j, row, k)

        return inner

    def manstitch(self, i, j, row, k):
        # function to manually stitch images together
        # accepts same as in man_masks
        # opens mspaint with concatenated images, user must place them correctly and save
        # program then replaces the old version for the edited version
        self.roo = (
            i,
            j,
            row,
            k,
        )  # line for debugging, to see what werethe parameters at the time of call
        path = self.stabdir / "masks"
        stitch_path = path / "_tmp.png"  # where to make the temporary edited file
        self.opath = f'{self.stabdir / "masks" / row.group}_{row.id}_{row.age}_{k}_m.png'  # where to put the final file
        apath = f"{self.stabdir / row.group}_{row.id}_{row.age}_{self.view_zooms[0]}.png"  # where to pull the first part from
        bpath = f"{self.stabdir / row.group}_{row.id}_{row.age}_{self.view_zooms[1]}.png"  # ... second file from
        ims = [
            np.array(Image.open(x)) for x in [apath, bpath] if Path(x).exists()
        ]  # only pass on paths that exist (filters out eg. when there is only one part)
        if len(ims) != 0:
            ims = np.concatenate(ims, axis=0)
        Image.fromarray(ims).save(stitch_path)
        subprocess.Popen(["mspaint.exe", Path(stitch_path)])
        answer = tk.messagebox.askokcancel(
            "Question", "When you are finished with paint, save and then press OK."
        )
        if not answer:
            self.del_tmps()  # remove any temporary files made in the process
            return
        self.nimage = Image.open(stitch_path)
        self.nimage.save(self.opath)  # save edited file in final path
        fishdisp = filtl(headtail("fishdisplay"), self.stabi.keys())[
            0
        ]  # create displayable image and get measurement
        extension = "_m" * ("_m" in fishdisp)
        self.rows[i][1][j].configure(
            image=PhotoImage(Image.fromarray(piximg(np.array(self.nimage), 3)))
        )
        stabirow = np.where(self.stabi.label == row.label)[0][0]
        self.stabi = set_df_element(
            self.stabi, row=stabirow, col="X2s" + extension, val=np.array(self.nimage)
        )
        self.stabi = set_df_element(
            self.stabi,
            row=stabirow,
            col="fishdisplay" + extension,
            val=find_length(np.array(self.nimage), prep="fish", show=1, retshow=1),
        )
        fishdisp_ind = list(row.keys()).index("fishdisplay" + extension)
        self.rows[i][1][fishdisp_ind].configure(
            image=PhotoImage(
                Image.fromarray(
                    piximg(
                        find_length(
                            np.array(self.nimage), prep="fish", show=1, retshow=1
                        ),
                        3,
                    )
                )
            )
        )
        if "fishlength" not in self.stabi.keys():
            self.stabi["fishlength"] = None
        self.stabi.loc[stabirow, "fishlength"] = find_length(
            np.array(self.nimage), prep="fish"
        )
        # save stabi data in stabi.npy and delete temporary files created in the process
        self.save_data()
        self.del_tmps()

    def del_tmps(self):
        # function to remove any _tmp files created during a manual editing process
        [
            os.remove(x)
            for x in flat_dir(self.stabdir)
            if headtail("", "_tmp.png")(Path(x).name)
        ]

    def paint(self, i, j, row, k):
        extension = "_m" * (k.split("_m")[0] == k)
        path = self.stabdir / "masks"
        mask_path = path / "_tmp.png"
        if k == "fishdisplay":
            self.opath = (
                f'{self.stabdir / "masks" / row.group}_{row.id}_{row.age}_X2s_m.png'
            )
        if k == "eyedisplay":
            self.opath = f"{self.stabdir / row.group}_{row.id}_{row.age}_X10_0.png"
        self.npath = (
            f'{self.stabdir / "masks" / row.group}_{row.id}_{row.age}_{k}_m.png'
        )
        Image.fromarray(np.array(Image.open(self.opath)) // 2 + 100).save(mask_path)
        subprocess.Popen(["mspaint.exe", Path(mask_path)])
        answer = tk.messagebox.askokcancel(
            "Question", "When you are finished with paint, save and then press OK."
        )
        if not answer:
            self.del_tmps()
            return
        self.nimage = Image.open(mask_path)
        ttimage = np.array(self.nimage)
        if len(ttimage.shape) < 3:
            ttimage = g2bgr(ttimage)
        nim = g2bgr(gstretch((bgr2g(ttimage) == 255) * 1))
        r = maxr_cont(bgr2g(nim), np.array(Image.open(self.opath)), show=1, retshow=1)
        rres = maxr_cont(bgr2g(nim), np.array(Image.open(self.opath)))
        self.rows[i][1][j].configure(image=PhotoImage(Image.fromarray(piximg(r, 3))))
        Image.fromarray(r).save(self.npath)
        stabirow = np.where(self.stabi.label == row.label)[0][0]
        self.stabi = set_df_element(self.stabi, row=stabirow, col=k, val=r)
        val_col = ["fishlength", "eyelength"][("eye" in k) * 1]
        self.stabi = set_df_element(self.stabi, row=stabirow, col=val_col, val=rres)
        self.save_data()

    def bipaint(self, i, j, row, k):
        print(i, j, row, k)
        extension = "_m" * (k.split("_m")[0] == k)
        path = self.stabdir / "masks"
        mask_path = path / "_tmp.png"
        self.npath = (
            f'{self.stabdir / "masks" / row.group}_{row.id}_{row.age}_lensdisplay_m.png'
        )
        self.opath = f"{self.stabdir / row.group}_{row.id}_{row.age}_X20_0.png"
        dispimage = ((np.array(Image.open(self.opath)) > self.x20_thresh) * 240).astype(
            np.uint8
        )
        Image.fromarray(dispimage).save(mask_path)
        subprocess.Popen(["mspaint.exe", Path(mask_path)])
        answer = tk.messagebox.askokcancel(
            "Question",
            "When you are finished with paint for positive mask, save and then press OK.",
        )
        if not answer:
            self.del_tmps()
            return
        posimage = np.array(Image.open(mask_path))
        if len(posimage.shape) > 2:
            posimage = bgr2g(posimage)
        posimage = posimage == 255
        Image.fromarray(dispimage).save(mask_path)
        subprocess.Popen(["mspaint.exe", Path(mask_path)])
        answer = tk.messagebox.askokcancel(
            "Question",
            "When you are finished with paint for negative mask, save and then press OK.",
        )
        if not answer:
            self.del_tmps()
            return
        negimage = np.array(Image.open(mask_path))
        if len(negimage.shape) > 2:
            negimage = bgr2g(negimage)
        negimage = 1 - (negimage == 255)
        mask = ((posimage * negimage) * 255).astype(np.uint8)
        stabirow = np.where(self.stabi.label == row.label)[0][0]
        todisplay = lens_prep(mask, show=1, retshow=1, todraw=dispimage)
        self.stabi = set_df_element(self.stabi, row=stabirow, col="lensmasks", val=mask)
        self.stabi = set_df_element(
            self.stabi, row=stabirow, col="lensdisplay", val=todisplay
        )
        self.rows[i][1][list(row.keys()).index(k)].configure(
            image=PhotoImage(Image.fromarray(piximg(todisplay, 3)))
        )
        if 'X20_1' in self.view_zooms:
            bulk.stabi = set_df_element(bulk.stabi, 'lensdisplayf', i, bulk.stabi.loc[i, 'X20_1'] * bulk.stabi.loc[i, 'lensmasks'].astype(bool))
            self.mkstabI()
            bulk.rows[i][1][[x for x in bulk.stabI.keys() if x not in ['age', 'group', 'id']].index('lensdisplayf')].configure(
    image=PhotoImage(Image.fromarray(piximg(bulk.stabi.loc[i, 'lensdisplayf'], 3)))
)
        self.save_data()

    def max_channels(self, path):
        # utility, isn't used during the program runtime
        mdims = 0
        for x in filter(headtail("", ".czi"), flat_dir(path)):
            cx = czi(x)
            ci = cx.dims.index("C")
            mdims = max(mdims, cx.size[ci])
        return mdims

    def ls2npims(self, x):
        if type(x) is str:
            x = self.stabdir / x
            if Path(x).exists():
                tmp = np.array(Image.open(x))
                if len(tmp.shape) > 2:
                    tmp = bgr2g(tmp)
                return tmp
        return None

    EMPTY_IMAGE = ctk.CTkImage(
        Image.fromarray(np.array([[255] * 250] * 125 + [[0] * 250] * 125))
    )

    def np2photo(self, x):
        if type(x) is np.ndarray:
            tmp = PhotoImage(Image.fromarray(piximg(x, 3)), master=self.nwin)
            return tmp
        return QWidget.EMPTY_IMAGE

    def mkresults(self):
        self.load_data(show=0)
        self.rtab = self.stabi[["label", "fishlength", "eyelength"]]
        self.rtab["fishlength"] *= QWidget.pxsize["X2"]
        self.rtab["eyelength"] *= QWidget.pxsize["X10"]
        self.rtab["eyearea"] = [
            (x > 0).sum() if type(x) is np.ndarray else 0
            for x in self.stabi["lensmasks"]
        ]
        thresh = tk.simpledialog.askfloat(
            "threshold eyes reflectance",
            "enter threshold value from 0 to 255 (0 is lowest possible value)",
        )
        self.rtab.loc[:, "eyethresh_ref"] = [
            ((x > 0) * self.stabi.loc[i, "X20_0"] > thresh).sum()
            if type(x) is np.ndarray
            else 0
            for i, x in enumerate(self.stabi["lensmasks"])
        ]
        thresh = tk.simpledialog.askfloat(
            "threshold eyes fluorescense",
            "enter threshold value from 0 to 255 (0 is lowest possible value)",
        )
        self.rtab.loc[:, "eyethresh_flu"] = [
            ((x > 0) * self.stabi.loc[i, "X20_1"] > thresh).sum()
            if type(x) is np.ndarray
            else 0
            for i, x in enumerate(self.stabi["lensmasks"])
        ]
        self.rtab[["age", "label", "group", "id"]] = self.stabi[
            ["age", "label", "group", "id"]
        ]
        self.rtab.to_csv(self.stabdir / "rtab.csv")

    def save_data(self):
        save_df(self.stabdir / "stabi", self.stabi)

    def load_data(self, show=1):
        stabidir = filedialog.askopenfilename(
            initialdir=".\\",
            title="Select A Data",
            filetypes=(("numpy files", "*.npy"), ("all files", "*.*")),
        )
        self.stabdir = Path(precede(stabidir))
        self.stabi = load_df(stabidir)
        if show:
            self.show_stab(fromstab=0)

    def show_thresh(self):
        thresh = tk.simpledialog.askfloat(
            "threshold eyes",
            "enter threshold value from 0 to 255 (0 is lowest possible value). for fluorescense enter nagative value.",
        )
        lab = 'X20_0' if (thresh >= 0) else 'X20_1'
        thresh = abs(thresh)
        nwin = tk.Toplevel()
        iframe = ctk.CTkScrollableFrame(nwin, width=2000, height=2000)
        pack_frames([iframe])
        rows = []
        thresholded = [
            (
                self.stabi.loc[i, "label"],
                PhotoImage(
                    Image.fromarray(
                        piximg(
                            ((x > 0) * self.stabi.loc[i, lab] > thresh) * 255, 1
                        ).astype(np.uint8)
                    ),
                    master=nwin,
                ),
            )
            if type(x) is np.ndarray
            else (self.stabi.loc[i, "label"], 0)
            for i, x in enumerate(self.stabi["lensmasks"])
        ]
        for i, row in enumerate(thresholded):
            rowframe = ctk.CTkFrame(iframe)
            rowlabs = [ctk.CTkLabel(rowframe, text=row[0])]
            if not type(row[1]) is int:
                rowlabs.append(
                    ctk.CTkLabel(rowframe, image=row[1], text="", height=160, width=160)
                )
            pack_frames([rowframe])
            pack_widgs(rowlabs)
            rows.append((rowframe, rowlabs))


# In[491]:


bulk = QWidget()


# In[493]:


if __name__ == "__main__":
    bulk.start()

