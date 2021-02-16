# Module for compatibility features

import os


def compat_no_algo():
    """ Compatibility to avoid error:
        tensorflow.python.framework.errors_impl.NotFoundError:  No algorithm worked! """
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def compat_lstm_cancel_error():
    """Setting this flag to True resolves a below error, while fitting LSTM model:
       tensorflow.python.framework.errors_impl.CancelledError:  [_Derived_]RecvAsync is cancelled."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
