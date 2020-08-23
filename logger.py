import sys
# import logging


class Logger(object):

    _instance = None

    def __init__(self, console_message, statusbar_message):
        Logger._instance = self

        # functions
        self.statusbar_message = statusbar_message
        self.console_message = console_message

    @classmethod
    def console_message(cls, text):
        if cls._instance is None:
            return

        cls._instance.console_message(text)

    @classmethod
    def status_message(cls, text):
        if cls._instance is None:
            return

        cls._instance.statusbar_message(text, 3000)

    @classmethod
    def message(cls, text):
        if cls._instance is None:
            return

        cls._instance.console_message(text)
        cls._instance.statusbar_message(text, 3000)


class Transcript:
    """Class used to redirect std output to pyqtconsole"""

    def __init__(self):
        self.terminal = sys.stdout

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        Logger.console_message(message)

    def flush(self):
        pass










