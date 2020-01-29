

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













