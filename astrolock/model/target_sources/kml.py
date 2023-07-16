import os.path
import zipfile
import xml.etree.ElementTree as ET
import astrolock.model.target_source as target_source
import astrolock.model.target as target
import astropy.coordinates

class KmlTargetSource(target_source.TargetSource):

    # so, like, if they release a new version, does my parser break for no reason?
    # i hate namespaces
    ns = {
        'kml': "http://www.opengis.net/kml/2.2"
    }

    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

        self.use_for_alignment = True

        self.filename = os.path.join('data', 'alignment_landmarks.kmz')

    def start(self):
        # todo: maybe follow the file to check for modifications?
        self.load_targets()

    def load_targets(self):
        if os.path.splitext(self.filename)[1] == '.kmz':
            with zipfile.ZipFile(self.filename, 'r') as kmz:
                with kmz.open('doc.kml', 'r') as kml:
                    xml = kml.read()
        else:
            with open(self.filename, 'r') as kml:
                xml = kml.read()

        self.target_map.clear()

        xml_root = ET.fromstring(xml)
        for placemark in xml_root.findall('.//kml:Placemark', KmlTargetSource.ns):
            display_name = placemark.find('kml:name', KmlTargetSource.ns).text
            point = placemark.find('kml:Point', KmlTargetSource.ns)
            coordinates = point.find('kml:coordinates', KmlTargetSource.ns).text
            lon_deg, lat_deg, alt_m = map(float, coordinates.split(','))

            new_target = target.Target.from_gps(lat_deg=lat_deg, lon_deg=lon_deg, alt_m=alt_m)
            new_target.display_name = display_name
            new_target.url = f'astrolock://kml/{display_name}'
            
            self.target_map[new_target.url] = new_target

