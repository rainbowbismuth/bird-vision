"""
Utility module to contain any code for any external libraries that get too noisy for my tastes.
"""


def silence_tensorflow():
    """Silence every warning of notice from tensorflow."""
    import tensorflow as tf
    import logging
    import os
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
