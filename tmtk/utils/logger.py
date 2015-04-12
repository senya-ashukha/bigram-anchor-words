import sys
import logging

def get_logger():
    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, ipdb

            traceback.print_exception(type, value, tb)
            ipdb.post_mortem(tb)

    sys.excepthook = info

    logging.basicConfig(
        format='%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    return logger