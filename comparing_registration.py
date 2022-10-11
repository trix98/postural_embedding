from skimage.io import imread
from scipy import ndimage
import os
from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import pandas as pd
import random
from scipy.ndimage.interpolation import rotate
from skimage.transform import EuclideanTransform
from skimage.transform import matrix_transform
from skimage import transform
from skimage import img_as_float
from sklearn.linear_model import LinearRegression
import seaborn as sns
from utils.orb_registration import EuclidORB
from utils.cluster_utils import manage_dir
from scipy.stats import shapiro, ranksums

seed =0


def parse_dlc(file_path='./data/dlc_pose/dlc_pose_estimation.h5'):
    """load DeepLabCut.h5 file and tidy up colnames"""

    data = pd.read_hdf(file_path)

    # tidy up the column names
    colnames = list(data.columns)
    names = []
    for i in colnames:
        names.append(i[1] + '_' + i[2])
    # print(names)
    data.columns = names

    return data


class EuclidDLC:

    # https://scikit-image.org/docs/stable/auto_examples/transform/plot_transform_types.html
    # https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.EuclideanTransform

    """using estimated (X, Y) px coordinates from DeepLabCut (DLC) for anatomical points for each frame
        Rigidly register each frame"""

    def __init__(self, data, labels, img_list, src_ix=0):
        self.et = EuclideanTransform()
        self.data = data
        self.labels = labels
        self.src_ix = src_ix
        print(labels)

        self.img_list = img_list
        self.src_coords = self.get_feat(self.src_ix)
        self.src_img = self.pre_prs(self.src_ix)

    def pre_prs(self, ix):
        """segment frame"""
        image_original = imread(self.img_list[ix])
        im_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(im_gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(gaus, 220, 255, cv2.THRESH_BINARY_INV)
        thresh1 = img_as_float(thresh1)
        return thresh1

    def get_feat(self, ix):
        """process (X, Y) px coordinates and round to get absolute coordinates for frame"""
        mx = np.zeros((len(self.labels), 2))
        for i, label in enumerate(self.labels):
            mx[i, 0], mx[i, 1] = self.data[label + '_x'][ix], self.data[label + '_y'][ix]
        return np.round(mx)

    def prepare_img(self, ix):
        """segment frame and get DLC coordinates"""
        dst_img = self.pre_prs(ix)
        dst_coords = self.get_feat(ix)
        return dst_coords, dst_img

    def transform_single(self, ix):
        dst_coords, dst_img = self.prepare_img(ix)

        # estimate by least squares rotation in radians and translation
        self.et.estimate(self.src_coords, dst_coords)
        # print(self.et.rotation, self.et.translation)

        tform = EuclideanTransform(rotation=self.et.rotation, translation=self.et.translation)
        tform = tform.params  # get transformation matrix as np.array
        # print(tf_mx)

        tf_img = transform.warp(dst_img, tform, cval=0)
        # plt.imshow(tf_img)
        tf_img = np.round(tf_img)

        return tf_img


def get_betas(fun, n_files):
    """get gradients for batch of segmented frames
    fun: registration function
    n_files: #frames
    """
    betas = []
    for i in range(n_files):
        fly = fun(i)
        coords = np.array([[x, y] for x, y in np.argwhere(fly == 1)])
        X = coords[:, 0].reshape(-1, 1)
        Y = coords[:, 1].reshape(-1, 1)
        reg = LinearRegression().fit(X, Y)
        betas.append(reg.coef_[0])
    return betas


class NoRotation:
    """segment but no registration"""

    def __init__(self, img_list):
        self.img_list = img_list

    def transform_single(self, ix):
        image_original = imread(self.img_list[ix])
        im_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(im_gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(gaus, 220, 255, cv2.THRESH_BINARY_INV)
        thresh1 = thresh1 / 255
        # thresh1 = img_as_float(thresh1)
        return thresh1


def sample_grid(fun, num_files, n=4):
    """plot random frames for given registration method"""
    plt.figure(figsize=(n, n))
    random.seed(seed)
    for i in range(n**2):
        plt.subplot(n,n,i+1)
        ix = random.randrange(0, num_files)
        img= fun(ix)*255
        img = np.stack((img,img, img), axis=2)
        plt.imshow(img)
        plt.axis("off")


def compare_sets(betas, save_dir):
    """
    violin plot to compare gradients for different regisration methods
    Shapiro-Wilk to test for normality
    Mann-Whitney U test to compare DLC and ORB
    """
    beta_df = pd.DataFrame(betas)
    beta_df = beta_df.astype(float)
    beta_stack = beta_df.stack().reset_index()
    beta_stack.columns = ['frame', 'method', 'value']
    #beta_stack['value'] = beta_stack['value'].astype(float)

    plt.figure()
    sns.violinplot(x='method', y="value", order=["No rotation", "DLC", "ORB"]
                   , data=beta_stack, palette="Blues", linewidith=1)
    plt.ylabel('Gradient', weight='bold')
    plt.xticks(rotation=20, weight='bold')
    plt.xlabel('')
    plt.savefig(save_dir + 'registration_distributions' + '.png', dpi=300, bbox_inches = "tight")

    print("="*40)
    print('Shapiro-Wilk test')
    print('No rotation', shapiro(beta_df['No rotation']))
    print('DLC', shapiro(beta_df['DLC']))
    print('ORB', shapiro(beta_df['ORB']))
    print('Mann-Whitney U test')
    print(ranksums(beta_df['DLC'], beta_df['ORB']))


if __name__ == '__main__':
    img_path = './data/images/'
    img_list = sorted(glob(img_path + '*' + '.png'), key=len)
    n_files = len(img_list)

    no_rotation = NoRotation(img_list=img_list)

    dlc_path = './data/dlc_pose/dlc_pose_estimation.h5'
    dlc_data = parse_dlc(file_path=dlc_path)
    labels = ['head_R', 'head_L', 'head_center', 'neck', 'abdomen', 'thorax']
    dlc_reg = EuclidDLC(data=dlc_data, labels=labels, img_list=img_list)

    orb_reg = EuclidORB(img_list=img_list, src_ix=0)

    functions = {'No rotation': no_rotation.transform_single, 'DLC': dlc_reg.transform_single,
                 'ORB': orb_reg.registration_comparison}
    save_dir = manage_dir(out_dir='results')
    beta_results = {}
    for name, func in functions.items():
        grads = get_betas(func, n_files=n_files)  # calculate gradients
        beta_results[name] = grads
        sample_grid(fun=func, num_files=n_files, n=7)  # plot same 49 frames for different methods
        plt.savefig(save_dir + name + '_sample.png', dpi=300, bbox_inches="tight")

    compare_sets(betas=beta_results, save_dir=save_dir)  # violin plots and statical analysis


