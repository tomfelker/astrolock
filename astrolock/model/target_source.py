class TargetSource:
    def __init__(self):
        self.observer_location = None
        self.observers = []
        self.target_map = {}
        self.use_for_alignment = False

    def notify_targets_updated(self):
        for observer in self.observers:
            observer.on_target_source_updated(self)

    def start(self):
        pass

    def stop(self):
        pass
    
