class PartialTileError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class NoTileError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class NoDataError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
