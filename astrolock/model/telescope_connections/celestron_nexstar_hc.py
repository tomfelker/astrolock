import serial
import serial.tools.list_ports

import astrolock.model.telescope_connections.com_port

class CelestronNexstarHCConnection(astrolock.model.telescope_connections.com_port.ComPortConnection):
    @staticmethod
    def get_url_scheme():
        return 'celestron_nexstar_hc:'

    def get_baud_rate(self):
        return 9600

    def __init__(self, url):
        super().__init__(url)