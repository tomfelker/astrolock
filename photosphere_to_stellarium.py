# see https://developers.google.com/streetview/spherical-metadata

# todo: reading image stuff
xmp = {}
xmp['FullPanoHeightPixels'] = 5120
xmp['CroppedAreaImageHeightPixels'] = 3798
xmp['CroppedAreaTopPixels'] = 642
xmp['PoseHeadingDegrees'] = 102

# the math stuff
landscape_ini = {}
landscape_ini['maptex_top'] = 90 - 180 * (xmp['CroppedAreaTopPixels'] / xmp['FullPanoHeightPixels'])
landscape_ini['maptex_bottom'] = 90 - 180 * ((xmp['CroppedAreaTopPixels'] + xmp['CroppedAreaImageHeightPixels']) / xmp['FullPanoHeightPixels'])
# this assumes the image wraps all the way around horizontally
landscape_ini['angle_rotatez'] = xmp['PoseHeadingDegrees'] + 90

# todo: file stuff
for k, v in landscape_ini.items():
	print(f'{k}={v}')
