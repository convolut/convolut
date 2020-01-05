import os
from .constants import ConsoleMode, TelegramMode, TensorboardMode, FlushType, LoaderName, ScheduleType, StateMode

GLOBAL_PREFIX = "CONVOLUT_"

# CRITERION
CRITERION_BCEDICE_EPS = float(os.environ.get(f"{GLOBAL_PREFIX}CRITERION_BCEDICE_EPS", 1e-7))

CRITERION_BCEDICE_THRESHOLD = os.environ.get(f"{GLOBAL_PREFIX}CRITERION_BCEDICE_THRESHOLD", None)
if CRITERION_BCEDICE_THRESHOLD:
    CRITERION_BCEDICE_THRESHOLD = float(CRITERION_BCEDICE_THRESHOLD)

CRITERION_BCEDICE_BCE_WEIGHT = float(os.environ.get(f"{GLOBAL_PREFIX}CRITERION_BCEDICE_BCE_WEIGHT", 0.5))
CRITERION_BCEDICE_DICE_WEIGHT = float(os.environ.get(f"{GLOBAL_PREFIX}CRITERION_BCEDICE_DICE_WEIGHT", 0.5))

CRITERION_DICE_EPS = float(os.environ.get(f"{GLOBAL_PREFIX}CRITERION_DICE_EPS", 1e-7))
CRITERION_DICE_THRESHOLD = os.environ.get(f"{GLOBAL_PREFIX}CRITERION_DICE_THRESHOLD", None)
if CRITERION_DICE_THRESHOLD:
    CRITERION_DICE_THRESHOLD = float(CRITERION_DICE_THRESHOLD)

# LOGGER
LOGGER_CONSOLE_MODE = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_CONSOLE_MODE", ConsoleMode.SingleLine)

LOGGER_FILE_FOLDER = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_FILE_FOLDER", "logs/file")
LOGGER_FILE_FILENAME = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_FILE_FILENAME", "log.txt")

LOGGER_TELEGRAM_TOKEN = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_TELEGRAM_TOKEN", None)
LOGGER_TELEGRAM_CHAT_ID = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_TELEGRAM_CHAT_ID", None)
LOGGER_TELEGRAM_MODE = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_TELEGRAM_MODE", TelegramMode.Basic)
LOGGER_TELEGRAM_PROXY = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_TELEGRAM_PROXY", 'https://api.telegram.org')

LOGGER_TENSORBOARD_MODE = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_TENSORBOARD_MODE", TensorboardMode.Basic)
LOGGER_TENSORBOARD_FOLDER = os.environ.get(f"{GLOBAL_PREFIX}LOGGER_TENSORBOARD_FOLDER", "logs/tensorboard")

# METRIC
METRIC_MANAGER_FLUSH_TYPE = os.environ.get(f"{GLOBAL_PREFIX}METRIC_MANAGER_FLUSH_TYPE", FlushType.PerEpoch)

# MODEL
MODEL_MANAGER_SCHEDULE_TYPE = os.environ.get(f"{GLOBAL_PREFIX}MODEL_MANAGER_SCHEDULE_TYPE", ScheduleType.PerEpoch)

# STATE
STATE_FILE_CHECKPOINT_FOLDER = os.environ.get(f"{GLOBAL_PREFIX}STATE_FILE_CHECKPOINT_FOLDER", "logs/checkpoints")
STATE_FILE_CHECKPOINT_SUFFIX = os.environ.get(f"{GLOBAL_PREFIX}STATE_FILE_CHECKPOINT_SUFFIX", "_checkpoint.pth")

STATE_MANAGER_STATE_MODE = os.environ.get(f"{GLOBAL_PREFIX}STATE_MANAGER_STATE_MODE", StateMode.Last)

# TRIGGER
TRIGGER_EARLY_STOPPER_WINDOW = int(os.environ.get(f"{GLOBAL_PREFIX}TRIGGER_EARLY_STOPPER_WINDOW", 3))
TRIGGER_EARLY_STOPPER_METRIC_NAME = os.environ.get(f"{GLOBAL_PREFIX}TRIGGER_EARLY_STOPPER_METRIC_NAME", "loss")
TRIGGER_EARLY_STOPPER_LOADER_NAME = os.environ.get(f"{GLOBAL_PREFIX}TRIGGER_EARLY_STOPPER_LOADER_NAME",
                                                   LoaderName.Valid)
TRIGGER_EARLY_STOPPER_DELTA = float(os.environ.get(f"{GLOBAL_PREFIX}TRIGGER_EARLY_STOPPER_DELTA", 1e-4))

RUNNER_EPOCHS = int(os.environ.get(f"{GLOBAL_PREFIX}RUNNER_EPOCHS", 1))

RUNNER_STEPS_PER_EPOCH = os.environ.get(f"{GLOBAL_PREFIX}RUNNER_STEPS_PER_EPOCH", None)
if RUNNER_STEPS_PER_EPOCH:
    RUNNER_STEPS_PER_EPOCH = int(RUNNER_STEPS_PER_EPOCH)

RUNNER_RESTART_ITERATOR = os.environ.get(f"{GLOBAL_PREFIX}RUNNER_RESTART_ITERATOR", None)
if RUNNER_RESTART_ITERATOR:
    RUNNER_RESTART_ITERATOR = RUNNER_RESTART_ITERATOR.lower() in ["true", "yes", "1"]
