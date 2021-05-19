import glob
from os.path import basename
from pathlib import Path
from PIL import Image

import cv2
import os

# filepaths
RESULTS_DIR = Path(r"C:\Users\machiel\OneDrive - Align Technology, Inc\Desktop\mid_train_results")
METHOD = "TO_VIDEO"
FPS = 6

def generate_vid():
    image_folder = str(RESULTS_DIR)
    video_name = str(RESULTS_DIR / (basename(RESULTS_DIR.parent) + ".avi"))
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, FPS, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def generate_gif():
    fp_in = RESULTS_DIR / "*.png"
    fp_out = RESULTS_DIR / (basename(RESULTS_DIR.parent) + ".gif")
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(str(fp_in)))]
    img.save(fp=str(fp_out), format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)

if __name__ == '__main__':
    if METHOD == "TO_VIDEO":
        generate_vid()
    else:
        generate_gif()
