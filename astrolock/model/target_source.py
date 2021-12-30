class TargetSource:
    def __init__(self):
        self.observer_location = None
        
    def get_targets(self):
        raise NotImplementedError

    def start(self):
        pass

    def stop(self):
        pass
    
