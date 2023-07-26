# AstroLock

This software will help you aim a telescope, particularly at moving targets like satellites, rockets, or planes, using a gamepad for precise input.  Alignment is computed within Astrolock, so you only need to point at some unidentified objects.  For satellites, Astrolock can download information about their orbits and track them automatically, with you fine-tuning using the joystick.  Or for cases when you can't predict the path in advance, it can track with momentum, so you only need joystick input when the target changes direction.

AstroLock is still rough around the edges, and you may need to dive into the Python code to get it working for you.

## Hardware

All you need is a telescope, a GoTo mount, a computer, and a gamepad.

Astrolock can currently talk to Celestron Nexstar hand controllers that control AltAz mounted telescopes.  (I'm using it with a CPC1100.)  It should be fairly easy to support EQ mounts, or to expand to other mount protocols - all that's needed is the ability to set the rates that each axis should move at, and to read the current position.

For testing, you can also connect to Stellarium with the Remote Control plugin.

PS4 or PS5 gamepads work, and it should be trivial to add support for other types.

I've been using Windows, but there's no reason it shouldn't work on Linux or Mac.

## Installation

There's no fancy packaging yet, so this is somewhat technical, yet straightforward.
1. get a working Python environment

1. clone this repository

1. install the requirements, which includes PyTorch (but no need for GPU support, the CPU is plenty fast)
    * `python -m pip install -r requirements.txt`

1. From the root of the cloned repository, run `python -m astrolock` 
    * or, run from Visual Studio Code - you'll probably want to edit some stuff anyway, until the GUI is more complete.

## Quick Start

1. Connect everything up: Telescope on, telescope connected to hand controller, hand controller connected to PC via USB, and gamepad connected to PC.

1. Run AstroLock.

1. Under the Status tab, select `celestron_nexstar_hc:COM`n (it should auto-detect) from the dropdown and hit Start.  You should see a bunch of info below, including ideally the GPS info from the telescope.  (If your scope doesn't have GPS, you'll need to edit the default location in tracker.py - did I mention the GUI is incomplete?)

1. Play around with the gamepad - see below for details.  Right trigger or Start will stop any motion.

1. Go to the Alignment tab to start aligning the telescope.  Use your gamepad to point at a target, and once it's centered in your eyepiece or camera, press "Add Current Observation" (or press Cross on the gamepad).  Repeat this three or more times.  Now press "Perform Alignment" - after a few seconds, it should give you a solution, including identifying the targets you used and telling you how level your tripod was.

1. Once you're aligned, you can track targets!  Go to the Targets tab and find what you're interested in, double-click, and the telescope should slew to the target and begin tracking it.  You can now fine-tune using the joysticks.

## Gamepad Controls

The left joystick is for coarse adjustment, and right joystick is for fine adjustment.  You can control the speed using the bumpers - L1 reduces sensitivity for finer control, and R1 increases sensitivity so you can move faster.

The axes are mapped similar to non-inverted video game controls, so that if you have a telrad sight or a camera, the crosshair will move in the direction that you push the stick.  When you pull the left trigger (L2), it will switch to what you would see if looking through an SCT with a diagonal.  (Someday there will be a GUI for configuring this to match other setups.)

What actually happens with this input depends on the mode:

### Momentum Mode

In this mode, your stick inputs accelerate the motors - so if you leave the sticks alone, the motors will continue at their present speed.  This mode works fine even if the telescope hasn't been aligned.  I've used it to track ISS and a Falcon 9 launch with some success, though it's a bit fidgety - one drawback is that when the target gets higher, you'll need to manually accelerate the azimuth.

### Target mode

When you've selected a target, you're in this mode.  The telescope will automatically try to follow the target, and your stick inputs are just controlling an offset - so, if you leave the sticks alone, the target should stay still in your view.

The offset includes a "lead time" - so for example, suppose the target is moving slowly near the horizon, and due to some problem with your clock, it's three seconds ahead of where AstroLock thought it would be.  All you need to do is use the sticks to center the target, and the offset in the direction that it's moving will be converted to a lead time and stored.  Now, if that was the only error and you do nothing more with the sticks, it should track the target accurately even when it's nearly overhead and the three-second error translates to a much larger angular error.

#### Spiral Search

What if you have the target centered as well as you can with your sights, but you can't see it in the eyepiece or camera?  Press and hold DPad-Right, and the telescope will move in a spiral pattern designed to cover all the nearby sky.  (Currently you need to edit the source code so it knows your field of view - GUI TODO.)  If you caught a glimpse of the target but you lost it, you can reverse the sweep with DPad-Left.  You can then fine-tune with the joysticks as usual.  Once you are centered, press Cross to reset the spiral, so that subsequent spirals will be centered on your current aim point.

## Details

### Telescope connections

#### Nexstar

The Nexstar telescopes have a bunch of units connected together talking via RS-232 - the hand controller, each motor controller, the GPS, and various autoguiding stuff.  When you plug in the hand controller via USB, it gives you a software serial port where you can talk directly to the hand controller, or ask the hand controller to forward messages to and from the other units - that's what we're using here.

One could use the hand controller to align the telescope, and then ask it to slew to AltAz coordinates, and let it do the math for the motor controllers.  But you can't directly set rates this way, or query the current position, or learn anything about the hand controller's pointing model, so it wouldn't work well if the tripod weren't completely level.  Hence, we're only using the hand controller to forward commands to and from the motor controllers, and biting off the challenge of doing alignment ourselves.  The motor controllers let us directly set axis rates and query the current values.

One limitation is speed - the serial connection is only 9600 bits/second, and so our loop of asking both motor controllers for their position and setting their speeds takes ~150 ms.  It might be possible to improve this by connecting directly to the serial interface via the PC port (which supposedly runs at 19200 bits/second), but that would require extra hardware.

Everything I know about this protocol I learned in [this PDF](http://www.paquettefamily.ca/nexstar/NexStar_AUX_Commands_10.pdf) - many thanks to Andre Paquette!

#### Stellarium

To connect, all you need to do is run Stellarium and enable the Remote Control plugin, with the default port of 8090 and no authentication.  Then you can select `stellarium:\\localhost:8090` form the Status tab, and hit Start to connect.  (TODO: support choosing other servers to connect to, and support authentication.)

When connected, we read the time from Stellarium, so you can use its time controls to simulate an upcoming satellite pass you'd like to observe later for real.  We don't currently (TODO) read the location, so you'll need to manually ensure the locations match.

The protocol is really simple, it just sends us JSON data over HTTP.  There are some rough edges:
* It lets us slew the view at arbitrary rates, although unfortunately these rates are specified realtive to the current FOV - so if you zoom in or out in Stellarium, it will throw off the tracking briefly, until we re-read the FOV.
* It won't actually tell us the apparent view angle, only the true angle, so there's some complicated problems related to refraction - upshot being, if we aim slightly lower than we ought to, that's why.
* There's a bug in the GMT time output, so reading Stellarium's time is only accurate to a second.  We do some fancy clamping to hopefully get subsecond accuracy and avoid big time jumps.
* Especially when moving, it's fairly slow, so the loop rate ends up being ~200 ms, a bit slower than the actual telescope.  This is a decent test, though.

### Alignment

The basic idea here is to enter a bunch of "observations", which are basically you saying "at this known time, the motor encoders had these known values, and the telescope was pointed at some unknown target.  AstroLock then has to do a bunch of math to determine a bunch of parameters which will later allow it to point in specified directions.

The motor encoders do not know their absolute position, only their position relative to where they were when the telescope was turned on.  So the most important parameters are the two encoder_offsets, which let us convert between the raw positions that the motor encoders can give, and the actual altitude and azimuth.  In theory, if the tripod were perfectly level and the observations perfectly accurate, those offsets could be determined with just one known star (which we don't implement) or two unknown stars (which we'll try to do, but I've never seen it work).

In practice, you should give AstroLock three unknown stars, and it can solve for not only the encoder offsets, but also the tilt of the tripod.  And you can give as many stars as you want, which lets AstroLock compute all of the following with increasing accuracy:

* The encoder offsets, necessary for even the roughest alignment
* The tilt of the tripod (zenith_pitch and zenith_roll).  As long as the tripod is level-ish to within a few degrees, with three or more points, we can recover this accurately.  (If it's not at all level, or the fork mount is on a wedge, see the code for a way to align without assuming anything at all.)
* Two errors related to mount inaccuracies.  One for if one fork arm is longer than the other, and one for if the telescope is aimed left or right of perpendicular to the altitude axis ("collimation" or "cone" error).
* A fudge factor for the refraction calculation.

Once you have your alignmnent points, just click "Perform Alignment" and wait a few seconds (perhaps staring at the console output).  It should identify all your observations, and show you the values of all those aprameters.  You can also see, for each observation, how far off it was from where AstroLock thinks it ought to be.  If one was particularly bad, you could disable it and try again.

You can also load and save alignments.  If you close AstroLock or lose your connection to the hand controller, you should be able to reconnect and load your alignment (or the autosaved one), "Perform Alignment" again, and be good to go - although if the hand controller lost power, it probably forgets the stepper offsets.  Saved alignments can also be useful for testing, i.e., seeing if adding some new factor to the model helps it achieve lower error.

### Target Sources

#### Skyfield

This uses the Skyfield library, JPL ephemerides, and NORAD / Celestrak TLE data to provide targets representing stars, planets, and satellites.

#### OpenSky

This uses the OpenSky API to grab ADS-B data and provide targets representing aircraft.

Note that the without an API key, the data is at least 10 seconds old (and there's a limit on how many queries you can do), so although we extrapolate it, it won't work well for planes that are manouvering.  It'd be cool to use an SDR to grab the ADS-B data ourselves with much less latency.

#### KML

This lets you use Google Earth (or anything that can output KML) to specify targets that are fixed to the Earth's surface.  If there are distant objects you can see at daytime and get accurate coordintes for, this would let you align your telescope during the day.

## Future Directions

### New tracking modes:

* Now that we can align the telescope, it would be possible to make a momentum mode that would remember the overall angular velocity, rather than the axis rates.  This way, the azimuth would automatically accelerate/decelerate, and the altitude would automatically slow down and reverse, when tracking a target moving in a great circle in the celestial sphere.

* We could also have momentum modes making other assumptions about the target's motion, fitting the recent history to those assumptions, and thus extrapolating.  For example, if we know the target is in a circular orbit at a given height, we could learn the other orbital parameters.  Or for a rocket launch, if we know the launch point and direction, we can guess the acceleration.

* For easy slewing, it'd be nice to have a non-momentum mode.  Mostly just needs a GUI to support switching modes.

* Once aligned, it'd be nice to have a mode that tracks the sideral rate.  This would help with choosing more alignment targets, or targeting stars you don't know the name of.

### GUI
Obviously the GUI needs a lot of work.  Small improvements:
* Want a way to show or select tracking modes - important if we add more.
* Better target selection - showing what stars are up, what satellites will be visible, etc.
* Needs improvements for connecting to a telscope, including settings for the connection
* Various settings: scope setup, fov, default location, etc.  Ideally with a non-boilerplatey way to add more settings.
* The Time tab needs love.
* Dark mode - I did battle with Tkinter for this but did not emerge victorious.  I can't find any reasonable way to learn the names of the myriad theme settings.

Or potentially large improvements, including switching to other toolkits (thinking PySide or DearPyGUI).

### Alignment Model

There are more things that would be fun to experiment with when aligning:
* Tube flexure
* Mirror flop
* Oval or off-center gears

### Threading

The code is already multithreaded, but Python's GIL being what it is, that doesn't help as much as it could.  Current issues are:
* Performing an alignment hangs the GUI for a long time - ideally it would either tick the GUI's main loop while running, or run in its own thread - supposedly PyTorch releases the GIL for long-running ops, so that may help.
* The telescope tracking loop runs in its own thread, but still, slow UI operations can hog the GIL and cause delays in the loop.
* Some of the target calculations are just very slow, on the order of milliseconds, so it might be useful to perform them on a different thread, somewhat ahead of the current time, and let the tracking loop just interpolate between those points.