from utils.processing_utils import extract_video_frames
from utils.orb_registration import EuclidORB
from comparing_registration import *


def process_all(img_path='./data/images/', save_dir='./data/prs_images', src_ix=0):
    """
    :param src_ix: Specify the frame to register to
    :param img_path: path to unprocessed video frames, frames must be saved as 0, 1, 2, 3, ...
    :param save_dir: str where to create directory and save processed frames to

    Process frames by first segmenting using binarisation
    Then rigidly registers all segmented frames to the frame specified by src_ix using
    matching key-points detected by Orientated FAST rotated BRIEF (ORB)
    """
    if os.path.exists(save_dir):
        return  # if the save_dir exists, the code bellow is not ran
    else:
        os.makedirs(save_dir)

    save_path = save_dir + '/{}.png'
    img_list = sorted(glob(img_path + '*' + '.png'), key=len)
    # instantiate classes for registration
    register = EuclidORB(img_list=img_list, src_ix=src_ix)
    for ix in trange(len(img_list), desc='processing images'):
        # process each frame in img_list and save
        img = register.transform_single(ix)
        cv2.imwrite(save_path.format(ix), img)


if __name__ == '__main__':

    vid_path = './data/video/fly_movie.avi'
    extract_video_frames('./data/images', vid_path)
    # print("frames:", len(os.listdir('./data/images')))

    process_all()  # perform segmentation and registration on all frames