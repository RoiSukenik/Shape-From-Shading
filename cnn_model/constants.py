import pathlib
CURRENT_DIR = str(pathlib.Path(__file__).parent.absolute())

""""sfs2_small_face_data sfs2_data sfs2_big_data sfs2_small_pics_data"""

"""PATHS & DATA"""
ZIP_NAME = "sfs2_small_pics_data"  # for loading and packaging
DATA_PATH_FOR_ZIP = CURRENT_DIR + "/data"
DATA_TO_TRAIN = CURRENT_DIR + "/" + ZIP_NAME
LOGS_DIR = str(pathlib.Path(__file__).parent.absolute()) + "/logs/"
DATA_NAME = "sfs"
DATA_PATH = f"/data/students/sfs/{ZIP_NAME}.zip"


"""IGNORED INDEX, RANDOM MORE"""


"""TRAINING CONSTS"""               # ORIGINAL VALS
MAX_DEPTH = 1.0                     # 1000.0
MIN_DEPTH = 0.0                     # 10.0
LEARNING_RATE = 0.00001       # 0.0001
VAL_RANGE = 1000.0                  # MAX_DEPTH / MIN_DEPTH
SSIM_WEIGHT = 0.1                     # 1.0
L1_WEIGHT = 1.0                  # 0.1
EPOCHS = 20

"""Optional VARS"""
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 1
SCHEDULER_GAMMA = 0.1

