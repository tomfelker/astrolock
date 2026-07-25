
2023-07-15

First test of it worked decently!

There was an ISS pass around 9pm... Stars only became visible a few minutes before, so it was a rush.

Alignment worked flawlessly.

Tracking was very very bouncy, because:
    - I had bumped up the proportional gain, shouldn't've done that without testing
    - the loop rate was horrible, ~500ms
        - global GUI updates didn't help
        - joystick updates came way to fast, so CPU busy

Saved the alignment data, "real_5x"

If you solve it without any mount errors, you get:

Step 5000 of 5000, loss 3.3238998753404303e-07, best angle loss was 0.0005980199784971774 with model 0
Done!
Final alignment:
AlignmentModel
        encoder_offsets: tensor([-97.4908, 323.9644]) deg
        zenith_roll: -0.3200315535068512 deg
        zenith_pitch: -0.683101236820221 deg
        non_perpendicular_axes_error: 0.0 deg
        collimation_error_in_azimuth: 0.0 deg


And with mount errors:

Step 5000 of 5000, loss 1.275354293284181e-07, best angle loss was 0.00043673202162608504 with model 0
Done!
Final alignment:
AlignmentModel
        encoder_offsets: tensor([-97.2970, 323.9524]) deg
        zenith_roll: -0.3099735379219055 deg
        zenith_pitch: -0.7118830680847168 deg
        non_perpendicular_axes_error: 0.048635825514793396 deg
        collimation_error_in_azimuth: 0.17704467475414276 deg

When setting up the tripod, using a digital level and a cell phone compass to approximate north, I measured:
around n:   e:
.7          .4
.01          .2 

so we're in the ballpark there

or maybe

2023-07-29, tracked ISS!
        worked very well, but I got on target late due to focusing
        video settings were:
                Exposure = 2ms
                Gain = 93
        and I would say it was underexposed by about half, and coulda been sharper image-wise
        ASI 'gain' values are centibel (1/10 decibel), something about voltage, something about 60 gain units to double...
        also I want to hit 183 so the vtech kicks in

        so: gain to 150, still 2 ms = plenty bright
        gain to 210, 1 ms = plenty bright (too bright?
        how about: gain 190, 1 ms

plan tonight:
        gain 190
        1 ms
        if too dim, bump gain
        if too bright, nerf exposure


2023-08-07
        Tried to track Tiangong.
                set the tripod intentionally unlevel by ~5 degrees - first time, this worked fine
                used focus.py to collimate, this worked awesome!
                realized Tiangong's path was such that I should rotate the tripod - so, rotated it and tried to realign
                this time it couldn't align... ran out of time before the pass
                tracked in momentum mode... I'm not good at this, only got a few frames
                also turns out I must've bumped the focus while moving the tripod - image was sort of strangely doubled

So, why didn't the alignment work?
        after many bugfixes, and switching to more consistent losses, a mystery remains:

        with limiting mag of 3, it always aligns to:

                Refining alignment: step 5000 of 5000: avg loss 5.609e-07, best loss 5.609e-07 with model 0
                Accuracy:
                        Observation 0, Sargas (HIP 86228), was off by 0.020519529112944693 deg
                        Observation 1, Almach (HIP 9640), was off by 0.055450347163059695 deg
                        Observation 2, Dubhe (HIP 54061), was off by 0.04504491748335559 deg
                Done!
                Final alignment:
                AlignmentModel
                        encoder_offsets: [119.591385  42.178425] deg
                        zenith_roll: 0.9328662763449895 deg
                        zenith_pitch: 10.943368361782158 deg
                        non_perpendicular_axes_error: 0.0 deg
                        collimation_error_in_azimuth: 0.0 deg
                        extra_refraction_coefficient: 0.0

        but with limiting mag of 2, it finds the right solution:

                Refining alignment: step 5000 of 5000: avg loss 1.195e-07, best loss 1.195e-07 with model 0
                Accuracy:
                        Observation 0, Arcturus (HIP 69673), was off by 0.009646177280478495 deg
                        Observation 1, Altair (HIP 97649), was off by 0.022387329051838645 deg
                        Observation 2, Vega (HIP 91262), was off by 0.024139679450450206 deg
                Done!
                Final alignment:
                AlignmentModel
                        encoder_offsets: [33.601837    0.18461345] deg
                        zenith_roll: -3.696126133501135 deg
                        zenith_pitch: -4.162482058519497 deg
                        non_perpendicular_axes_error: 0.0 deg
                        collimation_error_in_azimuth: 0.0 deg
                        extra_refraction_coefficient: 0.0

        the latter is clearly better (and I believe is correct...)

        so why isn't it found?


        also sometimes
                Refining alignment: step 5000 of 5000: avg loss 5.900e-07, best loss 5.900e-07 with model 0
                Accuracy:
                        Observation 0, MOON, was off by 0.056788013698590906 deg
                        Observation 1, MARS BARYCENTER, was off by 0.047697708929124984 deg
                        Observation 2, Fang (HIP 78265), was off by 0.017656877770389337 deg
                Done!
                Final alignment:
                AlignmentModel
                        encoder_offsets: [238.53256  58.26868] deg
                        zenith_roll: -2.6574699385878744 deg
                        zenith_pitch: -1.1510108857692667 deg
                        non_perpendicular_axes_error: 0.0 deg
                        collimation_error_in_azimuth: 0.0 deg
                        extra_refraction_coefficient: 0.0

        This persists even when:
                - setting stdev_zenith et al to 0
                - disabling below/above horizon filtering

        so, should be: [50, 104, 97]

        I guess the theory is, our algorithm (in unrolled form) is:
                - for each observation
                        - for each target
                                assume we were looking at that target during that observation
                                compute encoder offsets from there
                                based on that, find the stars we were looking at at the other times
                                optimize

        So the problem is, with more stars, we can make wrong assumptions about the other stars, unless we were
        very nearly correct about our zenith or other errors...

        Looking at where it fails exactly, when the magnitude limit is raised to 2.72, we include:
                Kaus Media (HIP 89931) 
                Tarazed (HIP 97278)
        and Tarazed is quite close to Altair, which is the correct one.  But Tarazed is not part of any of the bad solutions.
        So why does including it somehow disable the real solution?
        What's weird is - even in "full random" mode, this seems to occur...

        So, implemented brute_align, which does a brute force search over all combinations of stars... this is of course infeasible,
        so it actually has a bracket of the best ones found so far.  The algorithm is basically:

        for each new observation:
                in batches, combine each target with the previous bracket
                refine that a bit
                take the top k of those for the new bracket

        So then the question becomes, how big should the bracket be?  In my experiments, where I believe the correct assignment above
        is the lowest loss globally (haven't verified this)

Current Algorithms
        random_align():
                for some batches with random tripod tilts etc:
                        the batch index means:
                        for each target
                        for each observation
                                assume that observation was while pointing at that target
                                compute the raw angles that would imply
                                figure out full target assignments and losses
                                optionally refine the models
                        choose the best
New Algorithms
        so compute_best_loss_and_target just gives the one best, what if it gave top k?
        we could then test all perms of those k, O(k^obs)

        top_align():
                for random samples of tripod tilts and other model params
                        for each target
                        for each observation
                                assign encoder offsets based on assumption of pointing exactly at this target for this observation
                                for each other target
                                        find top k closest stars
                                        get permutations
                                                O((targets^2*observations*k^targets)


2026-07-08?

Tried to take the telescope out to record an ISS pass (the one from our example), but ran into issues:
- Couldn't connect to the cameras due to driver issues
- Laptop battery didn't last long, and inverter couldn't power it from the Celestron 17 power tank, it sagged too much under load and the inverter gave up.  Will use an extension cord next time.

2026-07-10

Tried to take the telescope out to record a Tiangong pass, but I ran into some issues, including a perf issue that precludes the whole thing from working.  When the real cameras are connected (and I've seen it even with the sim cameras), I occasionally get framerate hitches that take it almost to zero.  It seems progressively worse the longer it runs.  In Task Manager, I see the disk utilization spike to 100% when this happens.

Other issues:
- I had accidentally configured the wrong optic (well, left it at default), so when slewing to a target, it wouldn't work.  Did seem to work okay once I fixed that.
- Somehow, "follow target" defaulted to off.  (Or maybe it was off after clicking Stop Moving?), so at one point I selected a plane and it did nothing.
- My boresight was way off (I think the dovetail for the guide cam isn't sitting correctly, I need to hit it with a file).  This makes it hard to zero in because when you zoom the guide cam in, it zooms at the center, not at the main cam location.
- When the framerate dips while you're slewing, it can leave the scope slewing with no input.  Our manual slew commands should probably come with a timeout, so the backend can act as if you released the key even if the GUI goes AWOL.

Came back out later in the night, with the shared memory fixes:
- IR-pass filter cuts tons of light, and seeing was still horrible - maybe color would be better for the main cam.
- Boresight was way off, but using the red dot, I was able to point the main scope at what turned out to be saturn, and set boresight from there.  Saved 'tomu' and 'tomu2' settings files that should contain it.
- Was able to track two airplanes, worked pretty well (despite them being quite dim).  A bit sketchy-fast with this 55mm guide lens, but with the red dot I was able to hop ahead of them no problem for inital acquisition.  Once boresight was dialed in, no issues getting it in main frame.  Did bump the smoothing a bit, it was a little jerky, because of flasahing lights probably.
- Tried the focus thing - didn't work too well (couldnt' find the star, EMA seems to decay too much) - saved a big focus sweep of a star video for later experimentation.
- Saved some saturn shots.  Again, kinda lame without color, and seeing was awful.  What opposition amount are we?
- Recording still makes the UI a bit hangy (since it's back to reading .ser files).  Need to check if the files are full framerate though.  Maybe we should just always use shm, at least then UI can never block, at just the cost of a memcpy (or, well, we were gonna do that anyway...)
- Too hard to find the main vs guide cameras - need dividers, some minor UI design stuff
- seeing was just so horrible
- ended up leaving the scope and cameras outside, screw the dew, yolo.  zzz

2026-07-18

- after showing off the telescope to my friend, it was handy and there was a decent Tiangong pass, so tried to capture it - but...
- didn't set up early enough, and ran into a few problems:
- I tripped on a cord, pulling it out of the hand controller, and didn't notice, so thought I had a driver issue with the com port not showing up - rebooted - finally figured it out, but lost a few minutes
- because of this, didn't even have the main scope focused in time for the pass
- also, a recent 'hey claude please clean stuff up' led to it making a bug where it couldn't actually handle the click on the guide scope window, so it wouldn't have worked anyway.  syncing back was fine, and it figured it out.  I guess one needs better testing for all this, and earlier setup.
- worse still - it seems I broke the USB-micro connector off the hand controller board :-/
- the focus detector seems to lag - it ought to be grabbing latest, not next frames
- with the broken hand controller, still tried some stuff
- boresight is still pretty bad even after fixing the dovetail
- using rifle scope was necessary - it's much better than red dot for this purpose
- collimation thing kinda worked - focus is still very finicky though.
- should fix the crosshairs in the focus cam thing - need two crosshairs, and need collimation to be a circle not a dot so you can see the shape of the psf
- should do color display, now that we have it - shader magic I guess.  not sure if focus should though

2026-07-21
- gotta do better... plan is to look at an ISS and USA 60 pass late tonight, as sort of a test, and then be ready for Friday (2026-07-24) which has an 82 degree ISS pass.  Then, no more for 2 months (I hear it's roughly a 60 day cycle)
- So - after doing the star thing - will do some tests, but no more major changes.  This is the 0.99 release, after getting some pictures, we'll call it 1.0 and get some press.
- Also no optical changes to the scope - it's well collimated now (from the other day), so I'll not touch it.  Though it still needs some bugs cleaned off the secondary, and the corrector plate to be cleaned, and maybe one spec to brush off the primary...  for later.
- Also I'm not sure about the secondary/corrector clocking - years ago I had it off to fix a screw blocking sticker that had fallen on the primary, and may not have put it back right, as I didn't make a mark and had to guess from photos.  So, very interested to detect any astigmatism - but haven't really seen any thus far.

Still having perf issues getting both cameras to run at full framerate...
Got things dialed in, but couldn't see any planes... boresight is the biggest weak point here
maybe need spiral search?  Shouldn't be an issue on real stars, but kills it when zeroing on trees.
planes visible when near south san francisco at 11k feet from shed door

Haha, another learning:  focus between even very distant trees and infinity is so far, that is why I couldn't find the planes.  Even couldn't find the moon it was so far - like 5 turns of main focuser.

OKAY I AM ANNOYED - got all set up, tracked ISS fine - but, Main cam was flashing 'not recording',
and would not record with checking the Record now box, even tried disconnecting and reconnecting from it
so I saw it even on my screen, bits on this PC, but it didn't save the darn thing.  Le sigh...

2026-07-23

Tonight had a 33 degree ISS pass (would be higher but passed into shadow), so, used it as a dress rehearsal of tomorrow's 82 degree pass, the last good one for awhile.  I only allowed myself ~15 minutes to setup, which was too tight (even with the scope mostly ready to go).  In the event - I thought it didn't work - but, it actually did work!  Just at really low exposure, to the point that I didn't see it well myself.  Some notes, and a checklist:

Notes:
- telescope overshoots a bit on initial fix, which the sim doesn't do - hmm.  No huge issue though.
- tracking is a bit jiggly / noisy - smoothing to 2s helps a little bit, but maybe we just need more pixels
- however, the 25mm is much nicer to use than 55mm would be in terms of catching the target when not perfectly aimed.  without that, I probably would have missed tonight's pass, since I didn't have time to aim well.
- on dim targets, tracking can occasionally get stolen by nearby bright targets - we should have a tunable for this already, need to tweak it down and/or expose it
- hrm, .SER file doesn't encode any camera settings - we used to have, but no longer have, json sidecar file with per-frame info.
- I _think_ I was on default main camera settings, which is 1ms 190 gain.
- I occasionally see the background level on the main cam oscillate between lower and higher - not sure if that's a real thing (light pollution, someone's security camera hitting me with IR, etc), or some weird software issue.

Software TODO (after this pass, to avoid destabilizing...)
- more stars in UI (now that performance is fine) - or even constellation lines/names?
- change default lens to the 25mm one
- .SER file metadata
- expose tunable (and reduce it) for how much to trust target brightness vs distance from expected pos
- do we expose slew time horizon?  tuning that down might help overshoot / oscillations
- in the dark on this laptop, sometimes accidentally right click when I mean to click - perhaps ctrl-click or something would be better for unlocking
- pipper cleanup pass
- alignment settings should be higher up in the list to reflect settings order
- still having issues hitting full main cam bandwidth - even with guide cam disconnected, main cam set to 8 bit, not recording, CPU usage very low, main cam isn't hitting its full ~48 fps framerate.  (also, main detector is skipping frames - why is that slow?)  Tried main cam on the supposedly separate USB-C port which in theory has its own adapter.
- gamepad support (or arrow keys) for slewing would be useful, since trackpad slewing requires looking at the screen, so you can't also look at guidescope
- for boresighting - spiral search mode, and set boresight from auto track
- visual cue for overexposure - green, or maybe random colored, or maybe fixed pattern, highlights
- alignment?
- plate solving?


Checklist:
- telescope level, ideally pointing north, when turning it on
- run software, connect cameras, confirm these settings:
        - 25mm guide lens (todo: make default)
        - boresight (load settings, or manual - try (-.8, 1.2) mrad)

confirm main camera settings
        - 8-bit for main camera (play for framerate, though it still doesn't saturate)
        - 2 ms 400 gain (just a guess)
        - histogram on

- focus check on high star

confirm settings that might get changed when focusing on stars:
        - tracking to auto
        - auto-record to on on for both cameras

- tweak alignment to get stars to line up, confirm track, lock on star near track

- adjust tripod to tilt ~5 degrees west

2026-07-24

did exposure 2ms 130 gain + 2 stops - it was actually too bright, I was late to fix it...
also should maybe stop guide camera down a bit
tracking seemed to lag a bit, got out of frame a bit
but it mostly worked
tilting the tripod towards the pass worked, but did lose the middle
tomorrow night has a 55 degree pass, i'll try that for more sedate tracking and more fixes

here's the log:


(.venv) PS C:\projects\astrolock\.claude\worktrees\condescending-hofstadter-0b18b2> python -m astrolock.seeker.backend
[backend] session sessions\20260725T044209Z roles=['guide', 'main'] source=sky cmd_port=61790
[backend] TLE for 25544: downloaded (ISS (ZARYA), epoch 2026-07-24)
[backend] guide: ZWO ASI678MC + 8mm CS f/1.4 -> 51.282x30.219 deg, 103.132 arcsec/px, render 1920x1080 (bin 2x2)
[backend] main: ZWO ASI678MC + Celestron CPC 1100 -> 0.157x0.088 deg, 0.147 arcsec/px, render 3840x2160
[backend] note: main is color (RGGB) and unbinned but feeds detection; it'll be binned to mono for detection anyway -- --main-bin 2 halves bandwidth
[backend] reserved cores {'guide': [12, 13, 14, 15], 'main': [8, 9, 10, 11]}; backend+children on [0, 1, 2, 3, 4, 5, 6, 7]
[backend] detect roles: ['guide', 'main']
[backend] idle at startup: connect cams/mount in the GUI, or press Simulation -> Connect Simulated Cameras
[backend] sky_sim -> sessions\20260725T044209Z\20260725T044209Z_almanac.jsonl
[backend] sched: hero cam=- active tracker=-
[sky_sim] 15537 stars + 68 sat points, nav feed 93 stars + 9 bodies -> sessions\20260725T044209Z\20260725T044209Z_navigation.jsonl, epoch 2026-07-25T04:42:09.565Z [ISS (ZARYA) TLE epoch 2026-07-24, 0.7 d older than sim time]
[backend] mount selected: celestron_nexstar_hc:COM4 (press Connect)
[backend] mount connected: celestron_nexstar_hc:COM4
[backend] guide optics -> ['ZWO ASI678MM', '8mm CS f/1.4', None] (ok)
[backend] guide source -> zwo (disconnected; press Connect)
[backend] guide camera -> zwo:ZWO ASI678MM
[backend] capture started on guide
[backend] sched: hero cam=- active tracker=-
[cam] process priority: above
[cam] pinned to cores [12, 13, 14, 15]
[detect:guide] compute device: cuda
[cam] ZWO 'ZWO ASI678MM' 1920x1080 RAW16 12-bit auto-exposure WB=camera (MONO mosaic)
[cam] ZWO 'ZWO ASI678MM' controls @ connect (img_type=RAW16):
    AutoExpMaxExpMS        = 200   [1..60000]
    AutoExpMaxGain         = 400   [0..600]
    AutoExpTargetBrightness = 100   [50..160]
    BandWidth              = 40   [40..100]
    Exposure               = 2000   [32..2000000000]
    Flip                   = 0   [0..3]
    Gain                   = 190   [0..600]
    HardwareBin            = 0   [0..1]
    HighSpeedMode          = 0   [0..1]
    Offset                 = 3   [0..350]
    Temperature            = 149   (read-only)
[cam:guide] zwo 1920x1080 MONO 12-bit frame_limit=-1 file_limit=1 control=sessions\20260725T044209Z\control_guide_1.jsonl -> sessions\20260725T044209Z
W0724 21:42:48.567000 19164 Lib\site-packages\torch\utils\flop_counter.py:29] triton not found; flop counting will not work for triton kernels
[backend] record policy for guide: manual=False auto(while tracking)=False
[cam:guide] done, 59 frames total
[backend] guide bit depth = 8
[backend] sched: hero cam=- active tracker=-
[cam] process priority: above
[cam] pinned to cores [12, 13, 14, 15]
[cam] ZWO 'ZWO ASI678MM' 1920x1080 RAW8 auto-exposure WB=camera (MONO mosaic)
[cam] ZWO 'ZWO ASI678MM' controls @ connect (img_type=RAW8):
    AutoExpMaxExpMS        = 200   [1..60000]
    AutoExpMaxGain         = 400   [0..600]
    AutoExpTargetBrightness = 100   [50..160]
    BandWidth              = 40   [40..100]
    Exposure               = 2000   [32..2000000000]
    Flip                   = 0   [0..3]
    Gain                   = 190   [0..600]
    HardwareBin            = 0   [0..1]
    HighSpeedMode          = 0   [0..1]
    Offset                 = 3   [0..350]
    Temperature            = 158   (read-only)
[cam:guide] zwo 1920x1080 MONO 8-bit frame_limit=-1 file_limit=1 control=sessions\20260725T044209Z\control_guide_2.jsonl -> sessions\20260725T044209Z
[backend] main optics -> ['ZWO ASI678MC', 'Celestron CPC 1100', None] (ok)
[backend] main source -> zwo (disconnected; press Connect)
[backend] main camera -> zwo:ZWO ASI678MC
[backend] capture started on main
[backend] sched: hero cam=- active tracker=-
[detect:main] compute device: cuda
[cam] process priority: above
[cam] pinned to cores [8, 9, 10, 11]
[cam] ZWO 'ZWO ASI678MC' 3840x2160 RAW16 12-bit auto-exposure WB=neutral (BAYER_RGGB mosaic)
[cam] ZWO 'ZWO ASI678MC' controls @ connect (img_type=RAW16):
    AutoExpMaxExpMS        = 200   [1..60000]
    AutoExpMaxGain         = 400   [0..600]
    AutoExpTargetBrightness = 100   [50..160]
    BandWidth              = 100  (auto)   [40..100]
    Exposure               = 2000   [32..2000000000]
    Flip                   = 0   [0..3]
    Gain                   = 190   [0..600]
    HardwareBin            = 0   [0..1]
    HighSpeedMode          = 0   [0..1]
    MonoBin                = 0   [0..1]
    Offset                 = 3   [0..350]
    Temperature            = 171   (read-only)
    WB_B                   = 50   [1..99]
    WB_R                   = 50   [1..99]
[cam:main] zwo 3840x2160 BAYER_RGGB 12-bit frame_limit=-1 file_limit=1 control=sessions\20260725T044209Z\control_main_1.jsonl -> sessions\20260725T044209Z
[cam:main] done, 204 frames total
[backend] main bit depth = 8
[backend] sched: hero cam=- active tracker=-
[cam] process priority: above
[cam] pinned to cores [8, 9, 10, 11]
[cam] ZWO 'ZWO ASI678MC' 3840x2160 RAW8 auto-exposure WB=neutral (BAYER_RGGB mosaic)
[cam] ZWO 'ZWO ASI678MC' controls @ connect (img_type=RAW8):
    AutoExpMaxExpMS        = 200   [1..60000]
    AutoExpMaxGain         = 400   [0..600]
    AutoExpTargetBrightness = 100   [50..160]
    BandWidth              = 100  (auto)   [40..100]
    Exposure               = 2000   [32..2000000000]
    Flip                   = 0   [0..3]
    Gain                   = 190   [0..600]
    HardwareBin            = 0   [0..1]
    HighSpeedMode          = 0   [0..1]
    MonoBin                = 0   [0..1]
    Offset                 = 3   [0..350]
    Temperature            = 183   (read-only)
    WB_B                   = 50   [1..99]
    WB_R                   = 50   [1..99]
[cam:main] zwo 3840x2160 BAYER_RGGB 8-bit frame_limit=-1 file_limit=1 control=sessions\20260725T044209Z\control_main_2.jsonl -> sessions\20260725T044209Z
[cam:main] done, 2 frames total
[backend] main roi = 2048
[backend] sched: hero cam=- active tracker=-
[cam] process priority: above
[cam] pinned to cores [8, 9, 10, 11]
[cam] ZWO 'ZWO ASI678MC' 2048x2048 RAW8 auto-exposure WB=neutral (BAYER_RGGB mosaic)
[cam] ZWO 'ZWO ASI678MC' controls @ connect (img_type=RAW8):
    AutoExpMaxExpMS        = 200   [1..60000]
    AutoExpMaxGain         = 400   [0..600]
    AutoExpTargetBrightness = 100   [50..160]
    BandWidth              = 100  (auto)   [40..100]
    Exposure               = 2000   [32..2000000000]
    Flip                   = 0   [0..3]
    Gain                   = 190   [0..600]
    HardwareBin            = 0   [0..1]
    HighSpeedMode          = 0   [0..1]
    MonoBin                = 0   [0..1]
    Offset                 = 3   [0..350]
    Temperature            = 181   (read-only)
    WB_B                   = 50   [1..99]
    WB_R                   = 50   [1..99]
[cam:main] zwo 2048x2048 BAYER_RGGB 8-bit frame_limit=-1 file_limit=1 control=sessions\20260725T044209Z\control_main_3.jsonl -> sessions\20260725T044209Z
[backend] main control gain = 250.0 (live)
[backend] main control gain = 310.0 (live)
[backend] main control gain = 250.0 (live)
[backend] record policy for main: manual=False auto(while tracking)=False
[backend] guide optics -> ['ZWO ASI678MM', '25mm CCTV f/1.4 C-mount', None] (ok)
[backend] boresight -> (-0.100, 0.000) mrad
[backend] boresight -> (-0.200, 0.000) mrad
[backend] boresight -> (-0.300, 0.000) mrad
[backend] boresight -> (-0.400, 0.000) mrad
[backend] boresight -> (-0.500, 0.000) mrad
[backend] boresight -> (-0.600, 0.000) mrad
[backend] boresight -> (-0.700, 0.000) mrad
[backend] boresight -> (-0.800, 0.000) mrad
[backend] boresight -> (-0.800, -0.100) mrad
[backend] boresight -> (-0.800, -0.200) mrad
[backend] boresight -> (-0.800, -0.300) mrad
[backend] boresight -> (-0.800, -0.400) mrad
[backend] boresight -> (-0.800, -0.500) mrad
[backend] boresight -> (-0.800, -0.600) mrad
[backend] boresight -> (-0.800, -0.500) mrad
[backend] boresight -> (-0.800, -0.400) mrad
[backend] boresight -> (-0.800, -0.300) mrad
[backend] boresight -> (-0.800, -0.200) mrad
[backend] boresight -> (-0.800, -0.100) mrad
[backend] boresight -> (-0.800, -0.000) mrad
[backend] boresight -> (-0.800, 0.100) mrad
[backend] boresight -> (-0.800, 0.200) mrad
[backend] boresight -> (-0.800, 0.300) mrad
[backend] boresight -> (-0.800, 0.400) mrad
[backend] boresight -> (-0.800, 0.500) mrad
[backend] boresight -> (-0.800, 0.600) mrad
[backend] boresight -> (-0.800, 0.700) mrad
[backend] boresight -> (-0.800, 0.800) mrad
[backend] boresight -> (-0.800, 0.900) mrad
[backend] boresight -> (-0.800, 1.000) mrad
[backend] boresight -> (-0.800, 1.100) mrad
[backend] boresight -> (-0.800, 1.200) mrad
[backend] acquired target on guide at (951,483)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] sched: hero cam=guide active tracker=guide
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] alignment yaw -> -0.100 deg
[backend] alignment yaw -> -0.200 deg
[backend] alignment yaw -> -0.300 deg
[backend] alignment yaw -> -0.400 deg
[backend] alignment yaw -> -0.300 deg
[backend] alignment yaw -> -0.200 deg
[backend] alignment yaw -> -0.100 deg
[backend] alignment yaw -> -0.000 deg
[backend] alignment yaw -> +0.100 deg
[backend] alignment yaw -> +0.200 deg
[backend] alignment yaw -> +0.300 deg
[backend] alignment yaw -> +0.400 deg
[backend] alignment yaw -> +0.500 deg
[backend] alignment yaw -> +0.600 deg
[backend] alignment yaw -> +0.700 deg
[backend] alignment yaw -> +0.800 deg
[backend] alignment yaw -> +0.700 deg
[backend] focus started on main
[focus:main] compute device: cuda
[backend] sched: hero cam=- active tracker=-
[focus:main] processed 1228 frames
[backend] focus stopped
[backend] track source preference = guide
[backend] acquired target on guide at (973,89)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] sched: hero cam=guide active tracker=guide
[backend] lost target on guide
[backend] sched: hero cam=- active tracker=-
[backend] acquired target on guide at (970,435)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] sched: hero cam=guide active tracker=guide
[backend] guide control exposure = 4.0 (live)
[backend] guide control exposure = 8.0 (live)
[backend] sched: hero cam=- active tracker=-
[backend] acquired target on guide at (954,549)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] sched: hero cam=guide active tracker=guide
[backend] lost target on guide
[backend] sched: hero cam=- active tracker=-
[backend] acquired target on guide at (483,356)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] sched: hero cam=guide active tracker=guide
[backend] boresight -> (-0.800, 1.100) mrad
[backend] boresight -> (-0.800, 1.200) mrad
[backend] boresight -> (-0.800, 1.300) mrad
[backend] focus started on main
[focus:main] compute device: cuda
[focus:main] processed 5464 frames
[backend] focus stopped
[backend] sched: hero cam=- active tracker=-
[backend] track source preference = auto
[backend] record policy for main: manual=False auto(while tracking)=True
[backend] record policy for guide: manual=False auto(while tracking)=True
[backend] acquired target on guide at (1329,600)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] recording ON for guide
[backend] recording ON for main
[backend] sched: hero cam=guide active tracker=guide
[rec:main] -> recordings\20260725T045442664Z_main.ser
[rec:guide] -> recordings\20260725T045442669Z_guide.ser
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] handoff: now tracking on guide
[backend] sched: hero cam=guide active tracker=guide
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] handoff: now tracking on guide
[backend] sched: hero cam=guide active tracker=guide
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] handoff: now tracking on guide
[backend] sched: hero cam=guide active tracker=guide
[backend] coasting on guide (settled lock lost; holding last rate to re-acquire -- e-stop to halt)
[backend] recording off for guide
[backend] recording off for main
[rec:guide] done: 1768 frames written @ 19.2 fps (0 dropped: 0 lapped + 0 thinned; ~19.2 fps if none dropped) -> recordings\20260725T045442669Z_guide.ser
[backend] sched: hero cam=- active tracker=-
[rec:main] done: 5071 frames written @ 55.2 fps (0 dropped: 0 lapped + 0 thinned; ~55.2 fps if none dropped) -> recordings\20260725T045442664Z_main.ser
[backend] acquired target on guide at (1535,896)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] recording ON for guide
[backend] recording ON for main
[backend] sched: hero cam=guide active tracker=guide
[rec:main] -> recordings\20260725T045923860Z_main.ser
[rec:guide] -> recordings\20260725T045923911Z_guide.ser
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] handoff: now tracking on guide
[backend] sched: hero cam=guide active tracker=guide
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] handoff: now tracking on guide
[backend] sched: hero cam=guide active tracker=guide
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] main control gain = 190.0 (live)
[backend] main control gain = 130.0 (live)
[backend] handoff: now tracking on guide
[backend] sched: hero cam=guide active tracker=guide
[backend] coasting on guide (settled lock lost; holding last rate to re-acquire -- e-stop to halt)
[backend] recording off for guide
[backend] recording off for main
[backend] sched: hero cam=- active tracker=-
[rec:guide] done: 2718 frames written @ 19.2 fps (0 dropped: 0 lapped + 0 thinned; ~19.2 fps if none dropped) -> recordings\20260725T045923911Z_guide.ser
[rec:main] done: 7795 frames written @ 55.1 fps (0 dropped: 0 lapped + 0 thinned; ~55.1 fps if none dropped) -> recordings\20260725T045923860Z_main.ser
[backend] acquired target on guide at (1520,737)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] recording ON for guide
[backend] recording ON for main
[backend] sched: hero cam=guide active tracker=guide
[rec:main] -> recordings\20260725T050146005Z_main.ser
[rec:guide] -> recordings\20260725T050146020Z_guide.ser
[backend] lost target on guide
[backend] sched: hero cam=guide active tracker=-
[backend] recording off for guide
[backend] recording off for main
[rec:guide] done: 18 frames written @ 19.3 fps (0 dropped: 0 lapped + 0 thinned; ~19.3 fps if none dropped) -> recordings\20260725T050146020Z_guide.ser
[rec:main] done: 53 frames written @ 55.1 fps (0 dropped: 0 lapped + 0 thinned; ~55.1 fps if none dropped) -> recordings\20260725T050146005Z_main.ser
[backend] sched: hero cam=- active tracker=-
[backend] acquired target on guide at (896,530)px; will promote to main when it locks
[backend] track guide: sky: model GreatCircleModel, min-intercept 1.00s (position stiffness ~1.0/s), latency 0.00s, horizon 8.0s
[backend] recording ON for guide
[backend] recording ON for main
[backend] sched: hero cam=guide active tracker=guide
[rec:guide] -> recordings\20260725T050440969Z_guide.ser
[rec:main] -> recordings\20260725T050440976Z_main.ser
[backend] handoff: now tracking on main
[backend] sched: hero cam=main active tracker=main
[backend] recording off for guide
[backend] recording off for main
[rec:guide] done: 254 frames written @ 19.2 fps (0 dropped: 0 lapped + 0 thinned; ~19.2 fps if none dropped) -> recordings\20260725T050440969Z_guide.ser
[backend] sched: hero cam=- active tracker=-
[rec:main] done: 727 frames written @ 55.2 fps (0 dropped: 0 lapped + 0 thinned; ~55.2 fps if none dropped) -> recordings\20260725T050440976Z_main.ser
[backend] gui requested shutdown; stopping
[detect:guide] processed 27769 frames
[cam:guide] done, 27770 frames total
[cam:main] done, 77150 frames total
[detect:main] processed 64757 frames
[backend] removed session sessions\20260725T044209Z
[backend] done
(.venv) PS C:\projects\astrolock\.claude\worktrees\condescending-hofstadter-0b18b2> 
























