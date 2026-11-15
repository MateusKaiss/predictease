import os
import random

import numpy as np


def set_seed(seed: int, deep_learning: bool = False) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deep_learning:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
