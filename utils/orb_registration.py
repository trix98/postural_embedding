import cv2
import numpy as np
from skimage.io import imread
from skimage import transform, img_as_float
from skimage.transform import EuclideanTransform


class EuclidORB:
    """
    Class to process frames by segmentation and rigid registration (remove rotational and translational variance)
    Useful links to understand euclidean transformations in skimage
    https://scikit-image.org/docs/stable/auto_examples/transform/plot_transform_types.html
    https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.EuclideanTransform
    """

    def __init__(self, img_list, src_ix=0):
        """
        :param img_list: list of ordered directories to frames (each frame directory should end in /0.png, /1.png ....
        :param src_ix: index of img_list specifying frame to register all frames too
        """
        self.et = EuclideanTransform()  # instantiate skimage class used to estimate euclidean transformation matrix.
        self.src_ix = src_ix
        self.img_list = img_list

        #  self.img_list = sorted(glob(img_path + "*" + '.png'), key=len)
        self.dim = (imread(self.img_list[0]).shape[0], imread(self.img_list[0]).shape[1])
        self.src_orb = self.prs_src()  # process source image for registration

    def prs_orb(self, ix, thresh=75):
        """
        Process a frame to be used downstream in ORB keypoint detection.
        As the overall aim is to remove rotational and translational variance,
        a mask of the main body of the fly (head, thorax, abdomen) is created using thresholding
        and is used to insert the greyscale values of the fly that overlay with the mask into a blank array.
        Notably the wings and legs are not selected.

        For other datasets: A different threshold value or even technique may be necessary to segment the main body
        ORB has not been tested on registering multiple flies, if greyscale performs poorly, consider the red channel
        or alternative that may contain less information on the unique texture of the fly.

        :param ix: index for frame to process in img_list
        :param thresh: value for threshold - aim to selectively segment the body of the fly without appendages
        :return: an array where the main body of the fly take greyscale values, all else is white background
        """
        #  threshold out the main body of the fly to create a mask
        img = imread(self.img_list[ix])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(img_gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(gaus, thresh, 255, cv2.THRESH_BINARY_INV)

        #  copy greyscale frame pixels that mask with the main body of the fly onto an array with a white background
        coords = np.array([[x, y] for x, y in np.argwhere(thresh1 == 255)])  # get x, y coordinates
        blank = np.full((self.dim[0], self.dim[1]), 255, dtype='uint8')  # create white array
        for point in coords:  # inject each greyscale px onto blank that overlays with main body mask
            blank[point[0], point[1]] = img_gray[point[0], point[1]]
        return blank

    def segment(self, ix, value=220):
        """Segment the frame"""
        img = imread(self.img_list[ix])
        im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaus = cv2.GaussianBlur(im_gray, (5, 5), 0)
        ret, thresh1 = cv2.threshold(gaus, value, 255, cv2.THRESH_BINARY_INV)
        thresh1 = img_as_float(thresh1)
        return thresh1

    def prs_src(self):
        """
        Kp1, Keypoints (X, Y) px coordinates
        and
        des1, Description of keypoints
        for the source frame (specified by self.src_ix)
        """
        img = self.prs_orb(self.src_ix)
        # Initialize ORB.
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img, None)
        return kp1, des1

    def get_keypoints(self, dst_img):
        """
        Calculate 500 keypoints in dst (destination) frame
        Compare the descriptors of the keypoints with the source frame using Hamming distance
        select top 50% of matching keypoints

        :param dst_img:a processed frame (from self.prs_orb)
        :return: liist of (X,Y) coordinates of matching keypoints in dst (destination) and src (source)
        """
        list_kpts1, list_kpts2 = [], []

        # Initialize ORB.
        orb = cv2.ORB_create(nfeatures=500)
        # Detect and Compute for dst
        kp2, des2 = orb.detectAndCompute(dst_img, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(self.src_orb[1], des2, None)
        matches = list(matches)

        # Sort matches by score.
        matches.sort(key=lambda x: x.distance, reverse=False)
        # Retain only the top 50% of matches.
        numGoodMatches = int(len(matches) * 0.5)
        matches = matches[:numGoodMatches]

        for mat in matches:
            # Get the matching keypoints for each of the images.
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # Get the coordinates.
            (x1, y1) = self.src_orb[0][img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Append to each list.
            list_kpts1.append((int(x1), int(y1)))
            list_kpts2.append((int(x2), int(y2)))
        kpts1, kpts2 = np.asarray(list_kpts1), np.asarray(list_kpts2)
        return kpts1, kpts2

    def transform_single(self, ix):
        """
        Segment and register a singe frame specified by index in image list
        :param ix: index in img_list, the dst_image
        :return: The processed frame as an np.array
        """
        dst_img = self.prs_orb(ix)  # image for ORB comparison
        src_coords, dst_coords = self.get_keypoints(dst_img)

        # estimate by least squares rotation in radians and translation
        self.et.estimate(src_coords, dst_coords)
        # print(self.et.rotation, self.et.translation)

        tform = EuclideanTransform(rotation=self.et.rotation, translation=self.et.translation)  # Transformation matrix
        tform = tform.params  # get transformation matrix as np.array
        # print(tf_mx)

        dst_img_binary = self.segment(ix)

        tf_img = transform.warp(dst_img_binary, tform, cval=0)  #apply transformation matrix to dst
        # plt.imshow(tf_img)
        tf_img = np.round(tf_img) * 255

        return tf_img

    def registration_comparison(self, ix):
        img = self.transform_single(ix)
        img = np.round(img/255)
        return img
