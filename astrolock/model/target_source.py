class TargetSource:
    def __init__(self):
        self.observer_location = None
        self.targets_updated_callback = None
        
    def get_target_map(self):
        raise NotImplementedError

    def start(self):
        pass

    def stop(self):
        pass
    
