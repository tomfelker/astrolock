import astrolock.model.telescope_connection as telescope_connection

class DummyConnection(telescope_connection.TelescopeConnection):
    @staticmethod
    def get_url_scheme():
        return 'dummy:'

    @staticmethod
    def get_urls():
        return [ DummyConnection.get_url_scheme() ]

    def __init__(self, url):
        pass