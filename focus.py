import zwoasi
import pygame
import numpy as np
import torch
import os
import gc
import time
from collections import namedtuple

# Tunables!  (TODO: GUI)
exposure_usec = 100
gain_centibels = 0
ema_alpha = .02

star_crop_size = 64
expand_factor = 8
    

if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"

def lerp(a, b, t):
    return a * (1 - t) + b * t

def convert_linear_to_srgb(linear_rgb):
    _A = 0.055
    _PHI = 12.92
    _K0 = 0.04045
    _GAMMA = 2.4
    return torch.where(linear_rgb <= _K0 / _PHI, linear_rgb * _PHI, (1 + _A) * (linear_rgb**(1 / _GAMMA)) - _A)

def raw_to_bayer_planes(frame, debayer = True):
    if debayer:
        even_rows = frame[0::2, :]
        odd_rows = frame[1::2, :]

        upper_left = even_rows[:, 0::2]
        upper_right = even_rows[:, 1::2]
        lower_left = odd_rows[:, 0::2]
        lower_right = odd_rows[:, 1::2]

        frame = torch.stack([upper_left, upper_right, lower_left, lower_right], axis = -1)
    else:
        # still make a channel dimension, just keep it size 1
        frame = torch.unsqueeze(frame, dim=-1)

    return frame

def bayer_planes_to_subsampled_rgb(frame, debayer=True):
    if debayer:
        # todo: other bayer patterns
        red = frame[:, :, :, 0]
        green = (frame[:, :, :, 1] + frame[:, :, :, 2]) / 2
        blue = frame[:, :, :, 3]
        pygame_frame_whc = torch.stack([red, green, blue], axis = -1)
    else:
        white = frame[:, :, :, 0]
        pygame_frame_whc = torch.stack([white, white, white], axis = -1)

    return pygame_frame_whc

def center_of_mass_offset(image_bhwc, collapse_channels=True):
    """
    This returns the center of mass of the image, relative to its center, in units of pixels.
        input_bhwc is shape (batch, height, width, channels)
        return is shape (batch, dimension (y or x), channels)
    """

    if collapse_channels:
        image_bhwc = torch.sum(image_bhwc, dim=-1, keepdim=True)
    
    total_mass = torch.sum(image_bhwc, dim=(-3, -2), keepdim=True)
    # just to avoid div by zero
    if (total_mass == 0).any():
        total_mass += 1

    spatial_dims = (-3, -2)

    ret = []
    for dim in spatial_dims:
        dim_size = image_bhwc.shape[dim]
        multiplier = torch.linspace(-dim_size / 2.0, dim_size / 2.0, dim_size, device=dev)
        multiplier_shape = [1, 1, 1, 1]
        multiplier_shape[dim] = dim_size
        multiplier = torch.reshape(multiplier, multiplier_shape)
        moments = torch.multiply(image_bhwc, multiplier)
        com_in_dim = torch.sum(moments, dim=spatial_dims, keepdim=True) / total_mass
        com_in_dim = torch.squeeze(com_in_dim, axis=spatial_dims)
        ret.append(com_in_dim)
    ret = torch.stack(ret, dim=-2)
    return ret

def expand_image(image, expand_factor):
    return torch.repeat_interleave(torch.repeat_interleave(image, expand_factor, dim=-3), expand_factor, dim=-2)

FocusState = namedtuple('FocusState', ['star_crop_ema', 'focus_mean', 'focus_variance'])

def compute_focus(frame, state):
    # nerf  
    
    frame_max = torch.max(frame)
    star_mask = frame == frame_max

    frame_max = frame_max.cpu()

    com_offset = center_of_mass_offset(star_mask).cpu()
    #com_offset = torch.tensor([[[0], [0]]])

    image_size = torch.tensor(star_mask.shape[-3:-1], dtype=torch.int32)
    image_center = torch.tensor(image_size, dtype=torch.float32) / 2
    image_center_int = torch.tensor(image_center, dtype=torch.int32)
    # TODO: this is not respecting the batch dim, which is fine for now, but lame.
    star_center = image_center + com_offset[0, :, 0]
    star_center_int = torch.tensor(torch.trunc(star_center), dtype=torch.int32)

    star_mins = star_center_int - torch.tensor([star_crop_size // 2, star_crop_size // 2], dtype=torch.int32)
    star_maxs = star_mins + torch.tensor([star_crop_size, star_crop_size], dtype=torch.int32)
    
    if (star_mins < 0).any() or (star_maxs >= image_size).any():
        return state.focus_mean, frame, state
        
    star_crop = frame[:, star_mins[0]:star_maxs[0], star_mins[1]:star_maxs[1], :]

    if state.star_crop_ema is None:
        new_star_crop_ema = star_crop.clone()
    else:
        new_star_crop_ema = lerp(state.star_crop_ema, star_crop, ema_alpha)
    
    focus = frame_max
    new_focus_mean = lerp(state.focus_mean, focus, ema_alpha)
    new_focus_variance = lerp(state.focus_variance, (focus - state.focus_mean) * (focus - new_focus_mean), ema_alpha)

    star_crop_expanded = expand_image(star_crop, expand_factor)
    ema_expanded = expand_image(new_star_crop_ema, expand_factor)

    # this is what we will show in our window    
    combined = frame  # clone? for now, no need

    # add crosshairs to allow centering the star    
    combined[:, image_center_int[0], :, 0] = .2
    combined[:, :, image_center_int[1], 0] = .2

    # box around the star
    combined[:, star_mins[0], star_mins[1]:star_maxs[1], 0] = 1
    combined[:, star_maxs[0], star_mins[1]:star_maxs[1], 0] = 1
    combined[:, star_mins[0]:star_maxs[0], star_mins[1], 0] = 1
    combined[:, star_mins[0]:star_maxs[0], star_maxs[1], 0] = 1
    
    # blit the processed stuff into the corner
    inset_size = star_crop_size * expand_factor
    combined[:, 0:inset_size, 0:inset_size, :] = star_crop_expanded
    combined[:, inset_size:2 * inset_size, 0:inset_size, :] = ema_expanded

    new_state = FocusState(new_star_crop_ema, new_focus_mean, new_focus_variance)
    return new_focus_mean, combined, new_state

@torch.no_grad()
def process_frame(raw_frame_hw, state, debayer = True):
    # TODO: This is so dumb... torch can't support uint16 for whatever dumb reason... so need to profile to determine
    # which of the two dumb options is better: doing the conversion to float on the CPU in numpy, or doubling the memory
    # use on the CPU in numpy and then doing the divide in torch possibly on GPU if that's a thing.
    raw_frame_hw = np.array(raw_frame_hw, dtype=np.float32)

    frame = torch.tensor(raw_frame_hw, dtype = torch.float32, device=dev)
    frame /= (1 << 16) - 1
       
    # frame is now shape bhwc, 0 to 1

    frame = raw_to_bayer_planes(frame, debayer=debayer)

    # make batch dim
    frame = torch.unsqueeze(frame, dim=0)
    
    luckiness, frame, state = compute_focus(frame, state)

    frame = bayer_planes_to_subsampled_rgb(frame, debayer=debayer)
    
    pygame_frame_whc = convert_linear_to_srgb(frame)
    pygame_frame_whc *= 255.0
    pygame_frame_whc = torch.tensor(pygame_frame_whc, dtype=torch.uint8)
    pygame_frame_whc = torch.squeeze(pygame_frame_whc, axis = 0)
    pygame_frame_whc = torch.permute(pygame_frame_whc, dims=(1, 0, 2))

    audio = generate_audio(luckiness)

    return luckiness, pygame_frame_whc, audio, state

@torch.no_grad()
def generate_audio(luckiness):
    # tunables    
    audio_sample_rate = 48000.0
    audio_length_seconds = .1    
    base_freq_hz = 420.0
    envelope_octaves = 3
    luckiness_per_octave = .5
    octave_pattern = torch.tensor([0, 3], dtype = torch.float32) / 12.0
    #octave_pattern = tf.constant([0, 1, 2, 3, 4], dtype = tf.float32) / 5.0

    # shapes:
    # n means note (which of the various notes we're playing)
    # s means sample (which audio sample we're computing)
    # c means channels

    # derived constants
    audio_samples = int(audio_sample_rate * audio_length_seconds)
    freqs_octaves_n_array = []
    for octave in range(0, envelope_octaves + 1):
        freqs_octaves_n_array.append(octave_pattern + torch.tensor(octave, dtype=torch.float32))
    freqs_octaves_n = torch.concat(freqs_octaves_n_array, axis = 0)

    # dynamic stuff
    freqs_octaves_n += luckiness / luckiness_per_octave
    freqs_octaves_n = torch.fmod(freqs_octaves_n, envelope_octaves)
    
    freqs_hz_n = base_freq_hz * torch.pow(2.0, freqs_octaves_n)
    amplitudes_n = torch.maximum(torch.tensor(0.0, dtype=torch.float32), 1 - torch.abs((freqs_octaves_n / envelope_octaves) * 2 - 1))

    times_seconds_s = torch.linspace(0.0, audio_samples - 1.0, audio_samples, device=dev) / audio_sample_rate

    freqs_hz_ns = torch.unsqueeze(freqs_hz_n, dim=1).to(dev)
    amplitudes_ns = torch.unsqueeze(amplitudes_n, dim=1).to(dev)
    times_seconds_ns = torch.unsqueeze(times_seconds_s, dim=0)

    # nice sine waves - but, since sdlmixer sucks at crossfading, maybe want sawtooth wave or something so the cuts are less poppy
    #audio_ns = amplitudes_ns * tf.math.sin(times_seconds_ns * freqs_hz_ns * tf.constant(math.pi * 2, dtype=tf.float32))
    audio_ns = amplitudes_ns * torch.fmod(times_seconds_ns * freqs_hz_ns, 1.0)

    audio_s = torch.mean(audio_ns, dim=0)

    # do some magic so our chunk loops perfectly
    tile_fade_s = torch.abs(torch.linspace(-1.0, 1.0, audio_samples, device=dev))
    audio_s = audio_s * (1.0 - tile_fade_s) + torch.roll(audio_s, shifts=audio_samples // 2, dims=0) * tile_fade_s

    audio_int16_s = torch.tensor(audio_s * 32767.0, dtype = torch.int16)
    # it seems sdlmixer can't broadcast, need to stack for stereo
    #audio_int16_sc = tf.expand_dims(audio_int16_s, axis = 1)
    audio_int16_sc = torch.stack([audio_int16_s, audio_int16_s], axis = 1)

    return audio_int16_sc

zwoasi_library_file = os.getenv('ZWO_ASI_LIB') or 'c:/Program Files/ASIStudio/ASICamera2.dll'

try:
    zwoasi.init(zwoasi_library_file)
except:
    print('Set environment var ZWO_ASI_LIB to the path to ASICamera2.dll')
    exit()

cameras = zwoasi.list_cameras()

if len(cameras) == 0:
    print('No cameras found')
    exit()

camera = zwoasi.Camera(0)

camera_info = camera.get_camera_property()

print(camera_info)

camera.set_roi(image_type = zwoasi.ASI_IMG_RAW16)

camera.start_video_capture()

# seems to be usec
camera.set_control_value(zwoasi.ASI_EXPOSURE, exposure_usec)
# units are supposedly centibels of voltage gain, so add ~60 to double intensity.  (Experimentally, seems to need more like 80)
camera.set_control_value(zwoasi.ASI_GAIN, gain_centibels)

pygame.init()
display = pygame.display.set_mode(size = (1024, 768), depth = 32, flags = pygame.RESIZABLE)

# ug, this is stupid: need to jump through hoops to tell pygame and windows to just let me draw the pixels
# without scaling them up randomly since it's 2023 and everyone has 4k monitors.
import ctypes
try:
    ctypes.windll.user32.SetProcessDPIAware()
except AttributeError:
    pass

pygame.mixer.init(frequency = 48000, size = -16, channels = 2)

bit_depth = camera_info['BitDepth']

luckiness_cache = None

surface = None
running = True
sound = None
state = FocusState(None, 0.0, 0.0)
last_perf_time_ns = time.perf_counter_ns()
while running:
    
    perf_time_ns = time.perf_counter_ns()
    loop_time_ns = perf_time_ns - last_perf_time_ns
    last_perf_time_ns = perf_time_ns

    print(f"Loop time: {loop_time_ns * 1e-6:.1f} ms ({1 / (loop_time_ns * 1e-9):.1f} Hz)")

    try:
        frame = camera.capture_video_frame()
        #frame=torch.zeros((2160, 3840), dtype=torch.int16)
    except zwoasi.ZWO_IOError:
        print('timeout')
        continue
    
    process_start_ns = time.perf_counter_ns()
    luckiness, pygame_frame_whc, audio, state = process_frame(frame, state)
    process_time_ns = time.perf_counter_ns() - process_start_ns
    
    print(luckiness)
    print(f'Took {process_time_ns * 1e-6:.1f} ms')

    # tried crossfading, but SDL_mixer doesn't change volume smoothly, so it still has pops :-(
    new_sound = pygame.mixer.Sound(audio.cpu().numpy())
    new_sound.play(loops = -1)
    if sound is not None:
        sound.stop()
    sound = new_sound

    surface = pygame.surfarray.make_surface(pygame_frame_whc.cpu().numpy())
    display.blit(surface, (0, 0))
        
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
pygame.quit()
camera.close()
