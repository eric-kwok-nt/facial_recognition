import cv2
import logging
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert(vid_path: str, name: str, save_path: str, min_row=0, max_row=None, min_col=0, max_col=None):
    """Convert video to images

    Args:
        vid_path (str): Path to video
        name (str): Name of person
        save_path (str): Saving path
        min_row (int, optional): Cropping height start point. Defaults to 0.
        max_row ([type], optional): Cropping height end point. Defaults to None.
        min_col (int, optional): Cropping width start point. Defaults to 0.
        max_col ([type], optional): Cropping width end point. Defaults to None.
    """
    assert os.path.exists(vid_path), "Video does not exist!"
    assert os.path.exists(save_path), "Save path does not exist!"
    logger.info("Start conversion...")
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    max_row = max_row or image.shape[0] 
    max_col = max_col or image.shape[1]
    while success:
        image_path = os.path.join(save_path, f"{count}_{name}.jpg")
        cv2.imwrite(image_path, image[min_row:max_row, min_col:max_col])
        success, image = vidcap.read()
        count += 1
    
    logger.info(f"Converted {count} images")

if __name__ == '__main__':
    # vid_path = 'data/raw/videos/eric_lee.mp4'
    # name = 'eric_lee'
    # vid_path = 'data/raw/videos/eric_kwok.mp4'
    # name = 'eric_kwok'
    vid_path = 'data/raw/videos/clarence.mov'
    name = 'clarence'
    save_path = os.path.join('data/raw/images', name)
    # min_row = 250   # eric_lee
    # max_row = 1500  # eric_lee
    max_row = 600  # clarence
    min_col = 350 # clarence
    max_col = 900 # clarence
    # convert(vid_path, name, save_path, min_row=min_row, max_row=max_row)
    # convert(vid_path, name, save_path)
    convert(vid_path, name, save_path, max_row=max_row, min_col=min_col, max_col=max_col)
