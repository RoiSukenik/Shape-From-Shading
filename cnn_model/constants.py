import pathlib
import random
import torch

CURRENT_DIR = str(pathlib.Path(__file__).parent.absolute())

CUDA = True
NET_MIDSAVE_THREAD = True
SAVE_MODEL = True
GPU_TO_RUN = [0]

TEST = 1

""""sfs2_small_face_data sfs2_data sfs2_big_data sfs2_small_pics_data"""
"""PATHS & DATA"""
ZIP_NAME = "sfs4_data"  # for loading and packaging
if TEST:
    ZIP_NAME = "sfs2_small_face_data"
DATA_PATH_FOR_ZIP = CURRENT_DIR + "/data"
DATA_TO_TRAIN = CURRENT_DIR + "/" + ZIP_NAME
LOGS_DIR = str(pathlib.Path(__file__).parent.absolute()) + "/logs/"
DATA_NAME = "sfs"
DATA_PATH = str(pathlib.Path(__file__).parent.absolute()) + f"/{ZIP_NAME}.zip"
if TEST:
    DATA_PATH = f"/data/roeematan/sfs/{ZIP_NAME}.zip"

# DATA_PATH = f"/data/students/roeematan/sfs_data_30k.zip"

LS_AMOUNT = 4

"""TRAINING CONSTS"""  # ORIGINAL VALS
MAX_DEPTH = 1.0  # 1000.0
MIN_DEPTH = 0.0  # 10.0
LEARNING_RATE = 0.00001  # 0.0001
VAL_RANGE = 1000.0  # MAX_DEPTH / MIN_DEPTH
SSIM_WEIGHT = 1.0  # 1.0
L1_WEIGHT = 0.1  # 0.1
EPOCHS = 1
ACCUMULATION_STEPS = 16
MERGE_METHOD = "concat"
#MERGE_METHOD = "mean"

"""Optional VARS"""
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 3
SCHEDULER_GAMMA = 0.1
NOTES = ""  # NOTES FOR LOG
ADAPTIVE_LEARNER = True

"""
HYPER_PARAMS = {"LEARNING_RATE": LEARNING_RATE, "EPOCHS": EPOCHS, "SSIM_WEIGHT": SSIM_WEIGHT, "L1_WEIGHT": L1_WEIGHT,
                "USE_SCHEDULER": USE_SCHEDULER, "SCHEDULER_STEP_SIZE": SCHEDULER_STEP_SIZE,
                "SCHEDULER_GAMMA": SCHEDULER_GAMMA, "ACCUMULATION_STEPS": ACCUMULATION_STEPS,
                "ADAPTIVE_LEARNER": ADAPTIVE_LEARNER}
"""
HYPER_PARAMS = {"LEARNING_RATE": [0.000014],
                "EPOCHS": [40],
                "SSIM_WEIGHT": [1.0],
                "L1_WEIGHT": [0.1],
                "USE_SCHEDULER": [True],
                "SCHEDULER_STEP_SIZE": [3],
                "SCHEDULER_GAMMA": [0.1],
                "ACCUMULATION_STEPS": [10],
                "ADAPTIVE_LEARNER": [True]}

"""
RUNS = 80
EPOCHS_CONST = 40
if TEST:
    EPOCHS_CONST = 2
SSIM_WEIGHT_lst = [round(random.uniform(1.0, 1.1), 2) for i in range(RUNS)]
HYPER_PARAMS = {"LEARNING_RATE": [round(random.uniform(0.000001, 0.0001), 6) for i in range(RUNS)],
                "EPOCHS": [EPOCHS_CONST for i in range(RUNS)],
                "SSIM_WEIGHT": SSIM_WEIGHT_lst,
                "L1_WEIGHT": [round(1.11 - ssim_l, 2) for ssim_l in SSIM_WEIGHT_lst],
                "USE_SCHEDULER": [1-(True * (i % 5 == 0)) for i in range(RUNS)],
                "SCHEDULER_STEP_SIZE": [random.randint(1, 3) for i in range(RUNS)],
                "SCHEDULER_GAMMA": [round(random.uniform(0.01, 0.1), 2) for i in range(RUNS)],
                "ACCUMULATION_STEPS": [random.randint(4, 16) for i in range(RUNS)],
                "ADAPTIVE_LEARNER": [1-(True * (i % 2 == 0)) for i in range(RUNS)]}
"""
