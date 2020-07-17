import logging
import os



class LevelFilter(object):
    def __init__(self, level):
        self.level = level

    def filter(self, logRecord):
        return logRecord.levelno <= self.level


class RootLogger:
    def __init__(self, root_name, level=logging.DEBUG):
        self.root = logging.getLogger()
        # self.file_path = '../logs/{}/'.format(root_name)
        self.file_path = '{}/'.format(root_name)

        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)
        # self.baseformatter = logging.Formatter('%(asctime)s:%(module)s:%(funcName)s:%(message)s')
        self.baseformatter = logging.Formatter('%(asctime)s: %(levelname)s:[%(filename)s]: %(message)s')
        self.set_root(level=level)

    def set_root(self, level):
        self.root.setLevel(level=level)
        # log everything
        base_handler = logging.FileHandler('{}{}'.format(self.file_path,'ROOT.log'))
        base_handler.setFormatter(self.baseformatter)
        base_handler.setLevel(logging.DEBUG)

        # log only DEBUG
        debug_handler = logging.FileHandler('{}{}'.format(self.file_path,'DEBUG.log'))
        debug_handler.setFormatter(self.baseformatter)
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(LevelFilter(logging.DEBUG))

        # log only INFO
        info_handler = logging.FileHandler('{}{}'.format(self.file_path,'INFO.log'))
        info_handler.setFormatter(self.baseformatter)
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(LevelFilter(logging.INFO))

        # log only ERROR
        error_handler = logging.FileHandler('{}{}'.format(self.file_path,'ERROR.log'))
        error_handler.setFormatter(self.baseformatter)
        error_handler.setLevel(logging.WARNING)
        error_handler.addFilter(LevelFilter(logging.WARNING))

        # Stream Everything
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(self.baseformatter)
        stream_handler.setLevel(logging.DEBUG)

        self.root.addHandler(base_handler)
        self.root.addHandler(debug_handler)
        self.root.addHandler(info_handler)
        self.root.addHandler(error_handler)
        self.root.addHandler(stream_handler)

    def get_logger(self, logger_name, level=logging.DEBUG, build_file_handler=True):
        logger = self.root.getChild('{}'.format(logger_name))
        logger.setLevel(level)
        if build_file_handler:
            file_handler = logging.FileHandler('{}{}'.format(self.file_path,'{}.log'.format(logger_name)))
            file_handler.setFormatter(self.baseformatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        return logger

    def debug(self, message):
        self.root.debug(message)

    def info(self, message):
        self.root.info(message)

    def warn(self, message):
        self.root.warning(message)

    def error(self, message):
        self.root.exception(message)
