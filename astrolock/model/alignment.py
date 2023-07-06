import astropy.units as u
import torch

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
- 3 data points for unknown targets:
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

But, hmm - until we know stepper_offset[1], we actually _don't_ know angular distances between our measurements... 
    i.e., if our measurements differ in az, if alt was near horizon, that's a larger distance than if it was near the zenith


Algorithm:
    with T targets and N observations:
    
    compute directions for all targets at all observation times (shape: T, N, 3)

    for stepper_offset[1] ranging from 0 to 2pi step whatever:  (technically also ranges for the geometric errors, leaving only stepper_offset[0], azimuth_roll, and azimuth_pitch, which is just an unknown orientation)
        plug all raw values into the model, to get directions (shape: N, 3)
        for K = 0 to N-1
            we know the identities of the first K targets, so:
            compute dot products for all pairs of K targets (shape [K, K+1] with dot products 





"""

def dirs_to_mutual_angles(dirs):
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdims=True)
    num_dims = dirs.shape[-1]
    num_dirs = dirs.shape[-2]
    dir_indices = torch.arange(0, num_dirs)
    dir_indices_pairs = torch.cartesian_prod(dir_indices, dir_indices)
    from_indices = dir_indices_pairs[:, 0]
    to_indices = dir_indices_pairs[:, 1]
    num_indices = num_dirs * num_dirs
    from_indices = from_indices.reshape(num_indices, 1).expand(num_indices, num_dims)
    to_indices = to_indices.reshape(num_indices, 1).expand(num_indices, num_dims)
    from_dirs = torch.gather(dirs, -2, from_indices)
    to_dirs = torch.gather(dirs, -2, to_indices)
    dots = torch.linalg.vecdot(from_dirs, to_dirs, dim=-1)
    angles = torch.acos(torch.clamp(dots, -1.0, 1.0))
    angles = angles.reshape(num_dirs, num_dirs)
    return angles

def score_target_assignment(target_time_to_predicted_dir, time_to_observed_dir, time_to_target):
    num_assigned_targets = len(time_to_target)
    
    if num_assigned_targets < 2:
        return 0.0   

    #print(f"Trying {time_to_target}")

    # there's probably a more optimizable way to write this:
    time_to_predicted_dir = torch.stack([target_time_to_predicted_dir[target_index, time_index, :] for time_index, target_index in enumerate(time_to_target)])
    
    time_to_predicted_angles = dirs_to_mutual_angles(time_to_predicted_dir)
    time_to_observed_angles = dirs_to_mutual_angles(time_to_observed_dir[0:num_assigned_targets, :])

    error = torch.square(time_to_predicted_angles - time_to_observed_angles).mean().sqrt()

    print(f"Tried {time_to_target}, error {error}")

    return error

def rough_align_with_predictions(target_time_to_predicted_dir, time_to_observed_dir, known_time_to_target = [], known_time_to_target_error = 0.0):
    num_targets, num_observations, num_spatial_dimensions = target_time_to_predicted_dir.shape
    assert((num_observations, num_spatial_dimensions) == time_to_observed_dir.shape)
    num_known_targets = len(known_time_to_target)

    if num_known_targets < num_observations:
        incremental_time_to_target_and_error_pairs = []
        for target_index in range(num_targets):
            new_time_to_target = known_time_to_target.copy()
            new_time_to_target.append(target_index)
            new_error = score_target_assignment(target_time_to_predicted_dir, time_to_observed_dir, new_time_to_target)
            incremental_time_to_target_and_error_pairs.append((new_time_to_target, new_error))

        # we have to try all initial pairs, but after that, though, we can take only the top ones so far.
        if num_known_targets > 1:
            incremental_time_to_target_and_error_pairs = sorted(incremental_time_to_target_and_error_pairs, key = lambda pair: pair[1])
            incremental_time_to_target_and_error_pairs = incremental_time_to_target_and_error_pairs[0:num_observations]

        print(f"Best few so far is {incremental_time_to_target_and_error_pairs}")

        full_time_to_target_and_error_pairs = []
        for partial_time_to_target, partial_error in incremental_time_to_target_and_error_pairs:
            full_time_to_target_and_error_pair = rough_align_with_predictions(target_time_to_predicted_dir, time_to_observed_dir, partial_time_to_target, partial_error)
            full_time_to_target_and_error_pairs.append(full_time_to_target_and_error_pair)

        full_time_to_target_and_error_pairs = sorted(full_time_to_target_and_error_pairs, key = lambda pair: pair[1])
        best_full_time_to_target_and_error_pair = full_time_to_target_and_error_pairs[0]
        return best_full_time_to_target_and_error_pair
    else:
        return known_time_to_target, known_time_to_target_error
        #return known_time_to_target, score_target_assignment(target_time_to_predicted_dir, time_to_observed_dir, known_time_to_target)
    

def rough_align(target_time_to_dir, time_to_raw_axis_values, known_time_to_target = []):
    num_targets, num_observations, num_spatial_dimensions = target_time_to_dir.shape
    num_observations, num_raw_values = time_to_raw_axis_values.shape

        
    


class AlignmentModel:
    def __init__(self):
        # see https://www.wildcard-innovations.com.au/geometric_mount_errors.html
        # for helpful pictures and names.

        # axis 0 is azimuth, yaw, or right ascension (the one which needs acceleration when the other is near 90 degrees)
        # axis 1 is altitude, pitch, or declination (which needs no compensation)
        self.stepper_offsets = [0.0, 0.0] * u.rad
        self.azimuth_roll = 0 * u.rad
        self.azimuth_pitch = 0 * u.rad
        self.non_perpendicular_axes_error = 0 * u.rad
        self.collimation_error_in_azimuth = 0 * u.rad

        # How does the telescope move?  Imagine it's sitting on the ground, pointed at the horizon to the north (looking along +x in the North East Down frame)
        # - rotate it around local Y by self.azimuth_pitch
        # - rotate it around local X by self.azimuth_roll
        # - rotate it around local Z by self.raw_axis_values[0] - self.stepper_offsets[0]  (azimuth)
        # - rotate it around local X by self.non_perpendicular_axes_error
        # - rotate it around local Y by self.raw_axis_values[1] - self.stepper_offsets[1]  (altitude)
        # - rotate it around local Z by self.collimation_error_in_azimuth

    def errors_to_matrix(self):
        pass


target_time_to_predicted_dir = torch.tensor([
    [[.1, .2, .3], [.11, .2, .3], [.12, .2, .3]],  # first  target at .1 .2 .3, moving +x at .01 per sample
    [[.4, .5, .6], [.4, .51, .6], [.4, .52, .6]],  # second target at .4 .5 .6, moving +y at .01 per sample
    [[.7, .8, .9], [.7, .8, .91], [.7, .8, .92]],  # third  target at .7 .8 .9, moving +z at .01 per sample
    [[.1, .1, .1], [.1, .1, .2], [.1, .1, .3]],  # more bogus targets
    [[.2, .2, .2], [.2, .2, .3], [.2, .2, .4]],  # more bogus targets
])

time_to_observed_dir = torch.tensor([
    [.1, .2, .3],
    [.7, .8, .91],
    [.4, .52, .6]
])

time_to_target = torch.tensor([0, 2, 1])

error = score_target_assignment(target_time_to_predicted_dir, time_to_observed_dir, time_to_target)
print(error)

rough_align_result = rough_align_with_predictions(target_time_to_predicted_dir, time_to_observed_dir)
print(rough_align_result)
