import winrt
import winrt.windows.devices.geolocation as geolocation

async def on_position_changed(pos):
    print("woo")

async def foo():
    locator = geolocation.Geolocator()
    locator.desired_accuracy = geolocation.PositionAccuracy.HIGH
    #locator.desired_accuracy_in_meters = 0
    perms = await locator.request_access_async()
    
    pos = await locator.get_geoposition_async()

    #locator.add_position_changed(on_position_changed)
    return pos, locator

import asyncio
loop = asyncio.new_event_loop()
pos, locator = loop.run_until_complete(foo())
loop.close()

lat = pos.coordinate.latitude
lon = pos.coordinate.longitude


maplat = 37.565522
maplon = -122.0034402

print(f"error: lat: {maplat - lat}, lon: {maplon - lon}")


#url = "https://maps.google.com/?q=23.22,88.32&z=8"

#url = "http://maps.google.com/?z=12&q="

#url = "https://www.google.com/maps/place/magicplace/@"

#url = "https://www.google.com/maps/search/?api=1&query="
url = "https://www.google.com/maps/search/?api=1&query="
url += str(pos.coordinate.latitude)
url += ','
url += str(pos.coordinate.longitude)
#url += ',15z'

#url = f"https://earth.google.com/web/search/{lat},{lon}/"

print(url)

#import os
#os.system("start " + url)

