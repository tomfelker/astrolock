import serial
import serial.tools.list_ports
import threading

import astrolock.model.telescope_connection as telescope_connection

class ThreadedConnection(telescope_connection.TelescopeConnection):

    # subclasses should implement this, and loop until self.want_to_stop
    def loop(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)

    def start(self):
        self.want_to_stop = False
        self.loop_thread = threading.Thread(target = self.loop)
        self.loop_thread.start()
        return self


    def stop(self):
        self.want_to_stop = True
