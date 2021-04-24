import pathlib
CURRENT_DIR = str(pathlib.Path(__file__).parent.absolute())

"""PATHS & DATA"""
ZIP_NAME = "sfs2_small_data"  # for loading and packaging
DATA_PATH_FOR_ZIP = CURRENT_DIR + "/data"
DATA_TO_TRAIN = CURRENT_DIR + "/" + ZIP_NAME

"""TRAINING CONSTS"""
MAX_DEPTH = 1.0
MIN_DEPTH = 0.0
