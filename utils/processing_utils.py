import cv2
import os
import matplotlib.pyplot as plt


def extract_video_frames(raw_img_path, video_file):
    """
    :param raw_img_path: str to specify the directory to save the extracted frames
    :param video_file: path to video (.avi, .mp4), example './data/video/fly_movie.avi'

    if the raw_img_path does not exist,
    extract all frames and save to raw_img_path
    """

    if os.path.exists(raw_img_path):
        return  # if the dir exists, the code bellow is not ran
    else:
        os.makedirs(raw_img_path)

    save_path = raw_img_path + '/{}.png'
    cap = cv2.VideoCapture(video_file)
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        cv2.imwrite(save_path.format(count), frame)
        count += 1


def segment(img, value=220):
    """Segment frames by grayscale > gaussian blurring > thresholding"""
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(im_gray, (5, 5), 0)
    ret, thresh1 = cv2.threshold(gaus, value, 255, cv2.THRESH_BINARY_INV)
    return thresh1


def image_info(read_im, show=0):
    """helper function to get image properties"""
    un = np.unique(read_im)
    print("==============")
    print('max:', max(un), 'min:', min(un), "#unique", len(un))
    print('shape:', read_im.shape)
    if show:
        plt.imshow(read_im)
