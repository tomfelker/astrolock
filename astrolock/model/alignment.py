import astropy.units as u

class AlignmentDatum:
    def __init__(self, target, time, raw_axis_values):
        self.target = target
        self.time = time
        self.raw_axis_values = raw_axis_values

    def __repr__(self):
        return f'astrolock.model.alignment.AlignmentDatum({repr(self.target)}, {repr(self.time)}, {repr(self.raw_axis_values)})'
    

"""
Okay so how to do this?  Probably gonna use Tensorflow to paper over my lack of calculus knowledge...


What are the variables?

    - stepper offsets (2 numbers)
    - roll and pitch of azimuth axis (2 numbers, can assume zeros if you leveled/polar aligned it well)
    - axis misalignments (2 numbers, can assume zeros if telescope didn't fall off back of truck)
    - incorrect lattitude / longitude (2 numbers - for distant targets, degenerate with roll and pitch of azimuth axis)
    - incorrect time (1 number - for distant targets, degenerate with roll of azimuth)

What should be possible?
- 1 datum for a known target: can get stepper offsets
- 2 data points for known targets: can correct for imperfect leveling/polar alignment
- 2 data points for unknown targets:
    - should be able to deduce targets if accurate enough / not unlucky with targets of the same separation
    - technically possible you accidentally think you're upside down (only if both times are the same?)
- 3 data pointos for unknown targets:
    - much less chance of target misidentification

- 1 datum for a known target:
    - that gives 2 numbers
    - can solve for axis offsets (2 numbers)
        - need to assume everything else is perfect

So the real problem is identifying stars, O(S^N)
    - 1000 bright stars, to the three points...

Algorithm:
    given S stars and N observations (time and raw axis values):
    foreach star:
        foreach time:
            compute direction_to(star, time)

    foreach star_at_t0:
        foreach star_at_t1:
            compute angular distance between these observations
            find the top-n that match the distance between our measurements
    
    that's enough to establish our frame, then we can just keep adding nearest stars
    only problem is we're assuming the first two times are sufficiently accurate, if not, we may name the third stars wrongly...

    


"""

class AlignmentModel:
    def __init__(self):
        self.stepper_offsets = [0.0, 0.0] * u.rad
        pass

