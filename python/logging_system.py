import os
import logging
import datetime


class Logger(object):
    def __init__(self, level="DEBUG"):
        # Create a logger object
        current_date = datetime.date.today()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)
        self.project_name = "yolov9-tensorrt"
        self.project_path = self.get_project_path()
        self.log_dir = os.path.join(self.project_path, "logs")
        log_dir_name = "{}-{}-{}".format(current_date.year, current_date.month, current_date.day)
        self.log_dir = os.path.join(self.log_dir, log_dir_name)
        os.makedirs(self.log_dir, exist_ok=True)

    def get_project_path(self):
        script_path = os.path.abspath(__file__)
        path = os.path.dirname(script_path)
        pos = path.rfind(self.project_name)
        return os.path.join(path[:pos], self.project_name)

    def console_handler(self, level="DEBUG"):
        # Create a log processor for the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Add output format to processor
        console_handler.setFormatter(self.get_formatter()[0])

        # Return to controller
        return console_handler

    def file_handler(self, log_file, level="DEBUG"):
        log_file = os.path.join(self.log_dir, log_file)
        # Log processor for creating files
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)

        # Add output format to processor
        file_handler.setFormatter(self.get_formatter()[1])

        # Return to controller
        return file_handler

    def get_formatter(self):
        """Formatter"""
        console_fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s,%(funcName)s: %(message)s')
        file_fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s,%(funcName)s: %(message)s')
        # Returns a tuple
        return console_fmt, file_fmt

    def get_log(self, log_file, level="DEBUG"):
        # Adding a console processor to the logger
        self.logger.addHandler(self.console_handler(level))
        # Adding a file processor to the logger
        self.logger.addHandler(self.file_handler(log_file, level))

        # Return Log Instance Object
        return self.logger


if __name__ == "__main__":
    log = Logger()
    logger = log.get_log("log.txt")
    logger.info("hello world")