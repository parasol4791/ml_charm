# Module for compatibility features

def compat_no_algo():
    """ Compatibility to avoid error:
        tensorflow.python.framework.errors_impl.NotFoundError:  No algorithm worked! """
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
