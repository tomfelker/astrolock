class Target:
    def __init__(self):
        self.display_name = ''
        self.url = ''
        self.altaz_from_tracker = None
        self.last_known_location = None
        self.score = -float('inf')
        self.display_columns = {}

    # callers will do  target = target.updated_with(new_target)
    def updated_with(self, new_target):
        return new_target
    