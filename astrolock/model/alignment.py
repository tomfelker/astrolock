import math
import time
import astropy.units as u
import astropy.time
import numpy as np
import torch
import torch.nn
import itertools
from astrolock.model.util import *

class AlignmentSettings:
    def __init__(self):
        self.optimize_zenith_errors = True
        self.optimize_mount_errors = True
        # TODO: don't think this is correct until we can invert it also...
        self.optimize_refraction = False

        self.min_alt = -20.0 * u.deg
        self.max_alt = 89.0 * u.deg        
        self.num_batches = 100
        self.refine_during_search_steps = 1000
        self.full_random = True
        self.full_random_batch_size=5000
        self.final_refine_steps = 1000

        self.zenith_error_stdev_degrees = 6.0
        self.mount_errors_stdev_degrees = 0.1
        
        # TODO:
        #self.is_equatorial = False
        #self.latitude_rad = (45 * u.deg).to_value(u.rad)

class BruteAlignmentSettings:
    def __init__(self):
        self.max_batch_size = 10000
        #self.bracket_size = 500  # our favorite answer went away with 5 - todo, debug where it is
        self.refine_steps = 100 # our favorite model doesn't win 'till 700 or so


        self.bracket_quantile = .02

        self.optimize_zenith_errors = True
        self.optimize_mount_errors = False
        self.optimize_refraction = False

        self.debug_time_to_target = None
        # Arcturus, Altair, Vega
        # at limiting mag 2:
        #self.debug_time_to_target = [12, 21, 20]
        # at limiting mag 3:
        #self.debug_time_to_target = [50, 104, 97]

class AlignmentDatum:
    """
    The user will point the telescope at a star and click a button to collect these data.  Given a few of them, we can align the telescope.
    """

    def __init__(self, target, time, raw_axis_values):
        self.target = target
        self.time = time
        # axis 0 is azimuth, yaw, or right ascension (the one which needs acceleration when the other is near 90 degrees)
        # axis 1 is altitude, pitch, or declination (which needs no compensation)
        self.raw_axis_values = raw_axis_values

        # these are filled in by the alignment process, for GUI feedback on how close each point was.
        self.reconstructed_target = None
        self.angular_error = None
        self.loss = None        

    def __repr__(self):
        return f'astrolock.model.alignment.AlignmentDatum({repr(self.target)}, {repr(self.time)}, {repr(self.raw_axis_values)})'
    
    def to_json(self):
        return {
            "time": str(self.time.to_value('isot', 'date_hms')),
            "raw_axis_values": self.raw_axis_values.tolist()
        }

    @classmethod
    def from_json(cls, json_obj):
        return cls(
            target=None,
            time=astropy.time.Time(json_obj["time"]),
            raw_axis_values = np.array(json_obj["raw_axis_values"]) * u.rad
        )

"""
Okay so how to do this?

What are the variables?

    - encoder offsets (2 numbers)
    - roll and pitch of azimuth axis (2 numbers, can assume zeros if you leveled/polar aligned it well)
    - axis misalignments (2 numbers, can assume zeros if telescope didn't fall off back of truck)
    - incorrect lattitude / longitude (2 numbers - for distant targets, degenerate with roll and pitch of azimuth axis)
    - incorrect time (1 number - for distant targets, degenerate with roll of azimuth)
    - various fudge factors for mirror flop, 

What should be possible?
- 1 datum for a known target: can get encoder offsets
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

Here's a dead simple, low memory, embarassingly parallel method:
    - choose samples of the parameters we're interested in:
        - can be random so that we just choose how long to wait for a better solution
        - can easily make assumptions like "nearly level tripod"
        - can easily sample all parameters, not just orientation
    - for each sample:
        - transform observations into the real directions that would imply
        - for each real direction:
            dot product with each target  O(num_targets)
            take the best
            loss based on that

    whole thing is naiively O(acceptable_error^-1 * num_observations * num_targets)
        could probably throw a log in there with a better nearest neighbor search...
        but conversely, more targets means we probably need lower acceptable error
"""


def xyz_to_azalt(xyz):
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]    
    xy_len = torch.hypot(x, y)
    alt = torch.atan2(z, xy_len)
    az = torch.atan2(y, x)
    return torch.stack([az, alt], dim = -1)


def azalt_to_xyz(azalt):
    az = azalt[..., 0]
    alt = azalt[..., 1]
    cos_alt = torch.cos(alt)
    xyz = torch.stack([
        torch.cos(az) * cos_alt,
        torch.sin(az) * cos_alt,
        torch.sin(alt)
    ], dim=-1)
    return xyz


def wrap_to_pi(theta):
    ret = torch.atan2(torch.sin(theta), torch.cos(theta))
    return ret

def compute_dir_losses(predicted_dirs, observed_dirs):

    #losses = torch.sum((predicted_dirs - observed_dirs).square(), dim=-1, keepdim=False)    
    #losses = torch.sum((predicted_dirs - observed_dirs).square(), dim=-1, keepdim=False).square()
    losses = torch.sum((predicted_dirs - observed_dirs).square(), dim=-1, keepdim=False).sqrt()    
    return losses

def compute_model_losses(dir_losses, time_dim):
    model_losses = dir_losses.square().mean(dim=time_dim).sqrt()

    #model_losses = dir_losses.mean(dim=time_dim)
    #model_losses = dir_losses.square().mean(dim=time_dim).sqrt()
    #model_losses = dir_losses.square().mean(dim=time_dim).square()

    return model_losses

def compute_best_loss_and_target(batch_to_model, time_to_raw_axis_values, target_time_to_predicted_dir):
    # At each time, for each model, figure out where it thinks it was pointing (time_batch_to_observed_dir).
    time_batch_to_raw_axis_values = time_to_raw_axis_values.unsqueeze(1)
    time_batch_to_observed_dir = batch_to_model.dir_given_raw_axis_values(time_batch_to_raw_axis_values)

    # todo swizzle better?    
    target_time_batch_to_predicted_dir = target_time_to_predicted_dir.unsqueeze(2)
    target_time_batch_to_observed_dir = time_batch_to_observed_dir.unsqueeze(0)

    target_time_batch_to_loss = compute_dir_losses(target_time_batch_to_predicted_dir, target_time_batch_to_observed_dir)

    target_batch_time_to_loss = target_time_batch_to_loss.permute((0, 2, 1))

    # For each model at each time, find the dot of the angle to, and the index of, the target we think we were closest to.
    (batch_time_to_best_loss, batch_time_to_best_target) = torch.min(target_batch_time_to_loss, dim=0)

    return (batch_time_to_best_loss, batch_time_to_best_target)


def random_align(target_time_to_predicted_dir, time_to_raw_axis_values, settings):
    """
    Try many random guesses for alignment, and return the best one.  Hopefully, if it's good enough, it will be in the watershed of the global optimum.

    

    If num_batches is more than 1 (as it is by default), we will sample the default distribution of non-level tripods and non-perpendicular mount axes.

    TODO: Could support EQ mounts by initting zenith_pitch roughly to 90 deg minus latitude.
    """

    num_targets, num_observations, num_spatial_dimensions = target_time_to_predicted_dir.shape

    best_loss = math.inf
    best_time_to_target = None
    best_model = None

    optimize_zenith_errors = settings.optimize_zenith_errors and num_observations >= 3
    optimize_mount_errors = settings.optimize_mount_errors and num_observations >= 4

    if settings.full_random:
        # We will just randomize encoder_offsets, rather then computing ones good for each target/time.
        batch_size = settings.full_random_batch_size
        num_batches = settings.num_batches
    else:
        # Make a batch of models, one for each guess of the form "we were looking directly at this target index at this time index"
        batch_size = num_targets * num_observations
        if optimize_zenith_errors or optimize_mount_errors or settings.full_random:
            num_batches = settings.num_batches
        else:
            # if we're only checking our stepper offsets, then all the batches would be initialized the same anyway.
            num_batches = 1
        

    for batch in range(0, num_batches):
        batch_string = f'Evaluating {batch_size} models, batch {batch+1} of {num_batches}, '

        # First, we create a bunch of models, with parameters that are random (except possibly ensuring we are pointing directly at one single observation).

        batch_to_model = AlignmentModel((batch_size,))

        if optimize_zenith_errors:
            batch_to_model.randomize_zenith_error(np.deg2rad(settings.zenith_error_stdev_degrees))
        if optimize_mount_errors:
            batch_to_model.randomize_mount_errors(np.deg2rad(settings.mount_errors_stdev_degrees))

        if settings.full_random:
            batch_to_model.randomize_encoder_offsets()
        else:
            # This is copying our raw axis values (with known times but unknown target) into each target.
            target_time_to_raw_axis_values = time_to_raw_axis_values.unsqueeze(0).expand((num_targets, -1, -1))

            # Reshape these so we can jam them through the batched model.
            batch_to_raw_axis_values = target_time_to_raw_axis_values.reshape((batch_size, 2))
            batch_to_predicted_dir = target_time_to_predicted_dir.reshape(batch_size, num_spatial_dimensions)
            
            # For each element in the batch (each assumption about which target and time we were looking at), what would our encoder offsets have needed to be?
            with torch.no_grad():
                batch_to_model.encoder_offsets[:] = batch_to_raw_axis_values - batch_to_model.raw_axis_values_given_dir(batch_to_predicted_dir)

        # Okay, now our models are fully initialized.  Now figure out what stars each is closest to pointing to, and how close.
        (batch_time_to_best_loss, batch_time_to_best_target) = compute_best_loss_and_target(batch_to_model, time_to_raw_axis_values, target_time_to_predicted_dir)

        if settings.refine_during_search_steps > 0:
            print('')
            # Collect the directions of the nearest stars into the form needed for refinement
            batch_time_to_predicted_dir = target_time_to_predicted_dir[batch_time_to_best_target, torch.arange(num_observations)]
            time_modelbatch_to_raw_axis_values = time_to_raw_axis_values.unsqueeze(1)
            time_modelbatch_to_predicted_dir = batch_time_to_predicted_dir.swapaxes(0, 1)
            refine_alignment(batch_to_model, time_modelbatch_to_raw_axis_values, time_modelbatch_to_predicted_dir, num_steps=settings.refine_during_search_steps, settings=settings, debug_string_prefix=batch_string)

            # Since we changed the models, the angles to their closest stars have changed (hopefully, improved),
            # so we must redo this:
            (batch_time_to_best_loss, batch_time_to_best_target) = compute_best_loss_and_target(batch_to_model, time_to_raw_axis_values, target_time_to_predicted_dir)

            print('After refining, ', end='')
        else:
            print(batch_string, end='')

        batch_to_loss = compute_model_losses(batch_time_to_best_loss, time_dim=-1)

        # Get the loss of, and index of, the best model.
        (best_loss_in_batch, best_batch_index) = torch.min(batch_to_loss, dim=-1)

        # Extract that model from the batch and save it off if it's the best so far.
        if best_loss_in_batch < best_loss:
            print("new best", end='')
            best_loss = best_loss_in_batch
            best_time_to_target = batch_time_to_best_target[best_batch_index, :]
            best_model = AlignmentModel.choose_submodel(batch_to_model, best_batch_index)
        else:
            print("kept old", end='')

        print(f" loss {best_loss} with assignments {best_time_to_target}")

    return best_model, best_time_to_target

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch

def brute_align(target_time_to_predicted_dir, time_to_raw_axis_values, settings = BruteAlignmentSettings()):
    num_times, num_axes = time_to_raw_axis_values.shape
    num_targets, num_times, num_spatial_dimensions = target_time_to_predicted_dir.shape

    # this is our "bracket" holding the target assignments that are currently best.
    # although in the first iteration, it's size is num_targets rather than the bracket size
    prev_bracket_time_to_target = torch.arange(num_targets, dtype=torch.int64).unsqueeze(1)
    prev_bracket_to_model = None

    # adding each successive observation
    for current_step_num_times in range(2, num_times + 1):
        
        debug_partial_time_to_target = None
        if settings.debug_time_to_target is not None:
            debug_partial_time_to_target = torch.tensor(settings.debug_time_to_target[0:current_step_num_times], dtype=torch.int64)

        # Figure out how big the current bracket should be, so it includes the best settings.bracket_quantile of the assignments.
        prev_bracket_size = prev_bracket_time_to_target.shape[0]
        total_combinations = prev_bracket_size * num_targets
        bracket_size = int(math.ceil(total_combinations * settings.bracket_quantile))
        bracket_time_to_target = None

        print(f'Considering first {current_step_num_times} of {num_times} observations with a bracket size of {bracket_size}')

        assignment_pairs = itertools.product(range(prev_bracket_time_to_target.shape[0]), range(num_targets))
        for assignment_pairs_in_batch in batched(assignment_pairs, settings.max_batch_size):
            batch_size = len(assignment_pairs_in_batch)
            batch_time_to_target = torch.zeros(size=(batch_size, current_step_num_times), dtype=torch.int64)
            batch_to_model = AlignmentModel((batch_size,))

            print(f'Starting a batch of size {batch_size}')

            debug_batch_index = None
            for batch_index, (old_bracket_index, new_assignment) in enumerate(assignment_pairs_in_batch):
                old_assignments = prev_bracket_time_to_target[old_bracket_index]
                batch_time_to_target[batch_index, :current_step_num_times - 1] = old_assignments
                batch_time_to_target[batch_index, -1] = new_assignment
                if prev_bracket_to_model is not None:
                    # this means:
                    #batch_to_model[batch_index] = bracket_to_model[old_bracket_index]
                    batch_to_model.assign_submodel(batch_index, prev_bracket_to_model, old_bracket_index)
                #print(f'{batch_index}: {batch_time_to_target[batch_index]}')
                if debug_partial_time_to_target is not None and (batch_time_to_target[batch_index] == debug_partial_time_to_target).all():
                    debug_batch_index = batch_index
                    print(f'\tIncluding debug assignments {batch_time_to_target[batch_index]} at index {batch_index}')
        
            indices=batch_time_to_target.unsqueeze(-1).expand((-1, -1, num_spatial_dimensions))
            batch_time_to_predicted_dir = torch.gather(input=target_time_to_predicted_dir, dim=0, index=indices)

            if prev_bracket_to_model is None:
                # Init our batch models with the average of the encoder offsets they'd need to point at the already
                # assigned targets.  We could instead initialize with the model from the previous step, but that seems
                # a bit less symmetrical, and more complex to implement.
                with torch.no_grad():
                    for init_time in range(0, current_step_num_times):
                        raw_axis_values = time_to_raw_axis_values[init_time].unsqueeze(0)
                        batch_to_predicted_dir = batch_time_to_predicted_dir[:, init_time, :]
                        with torch.no_grad():
                            batch_to_model.encoder_offsets[:] += raw_axis_values - batch_to_model.raw_axis_values_given_dir(batch_to_predicted_dir)
                    batch_to_model.encoder_offsets[:] /= current_step_num_times
            

            # swizzle things a bit for refine_alignment()
            time_batch_to_raw_axis_values = time_to_raw_axis_values.unsqueeze(1)            
            time_batch_to_predicted_dir = batch_time_to_predicted_dir.permute((1, 0, 2))
            refine_alignment(batch_to_model, time_batch_to_raw_axis_values[0:current_step_num_times, :, :], time_batch_to_predicted_dir, num_steps=settings.refine_steps, settings=settings)

            # after refining, compute the losses
            time_batch_to_observed_dir = batch_to_model.dir_given_raw_axis_values(time_batch_to_raw_axis_values)
            batch_time_to_observed_dir = time_batch_to_observed_dir.permute((1, 0, 2))
            batch_time_to_loss = compute_dir_losses(batch_time_to_predicted_dir, batch_time_to_observed_dir[:, 0:current_step_num_times, :])
            batch_to_loss = compute_model_losses(batch_time_to_loss, time_dim=1)

            if debug_batch_index is not None:
                print(f'After refinement, debug assignment loss was {batch_to_loss[debug_batch_index]}, ({batch_time_to_loss[debug_batch_index]}), model: {AlignmentModel.choose_submodel(batch_to_model, debug_batch_index)}')
                debug_best_loss, debug_best_loss_index = torch.min(batch_to_loss, dim=0)
                print(f'vs best loss was model {debug_best_loss_index}, loss {batch_to_loss[debug_best_loss_index]}, ({batch_time_to_loss[debug_best_loss_index]}), model: {AlignmentModel.choose_submodel(batch_to_model, debug_best_loss_index)}')

            # Now merge this batch with the bracket
            if bracket_time_to_target is not None:
                merged_time_to_target = torch.cat((batch_time_to_target, bracket_time_to_target), dim=0)
                merged_to_loss = torch.cat((batch_to_loss, bracket_to_loss), dim=0)
                merged_to_model = AlignmentModel.concatenate_batches(batch_to_model, bracket_to_model)
            else:
                # for the first batch in each step, the bracket is empty
                merged_time_to_target = batch_time_to_target
                merged_to_loss = batch_to_loss
                merged_to_model = batch_to_model

            # pare the bracket back down to its proper size            
            best_losses, best_indices = merged_to_loss.topk(k=min(bracket_size, merged_to_loss.shape[0]), dim=0, largest=False)
            bracket_to_loss = best_losses
            bracket_time_to_target = merged_time_to_target[best_indices]
            bracket_to_model = AlignmentModel.choose_submodels(merged_to_model, best_indices)

            if debug_batch_index is not None:
                best_indices_equal_debug_batch_index = (best_indices == debug_batch_index)
                if best_indices_equal_debug_batch_index.any():
                    place = best_indices_equal_debug_batch_index.nonzero()[0].item() + 1
                    print(f'Debug assignment made the cut, {ordinal(place)} place out of {bracket_size}, in the top {place / total_combinations} fraction of all considered on this step')
                else:
                    print("Debug assignment didn't make the cut")

        prev_bracket_time_to_target = bracket_time_to_target
        prev_bracket_to_model = bracket_to_model


    # todo: maybe refine the final bracket further

    # now we're done, extract the one we like
    best_loss_in_bracket, best_bracket_index = torch.min(bracket_to_loss, dim=0)
    best_time_to_target = bracket_time_to_target[best_bracket_index]
    best_model = AlignmentModel.choose_submodel(bracket_to_model, best_bracket_index)

    print(f'Selected model {best_bracket_index} with loss {best_loss_in_bracket}')

    return best_model, best_time_to_target


def refine_alignment(model, time_modelbatch_to_raw_axis_values, time_modelbatch_to_predicted_dir, num_steps, settings, debug_string_prefix = ''):
    num_observations = time_modelbatch_to_raw_axis_values.shape[0]

    params_to_optimize = [model.encoder_offsets]

    if settings.optimize_zenith_errors and num_observations >= 2:
        params_to_optimize.append(model.zenith_pitch)
        params_to_optimize.append(model.zenith_roll)
    if settings.optimize_mount_errors and num_observations >= 3:
        params_to_optimize.append(model.collimation_error_in_azimuth)
        params_to_optimize.append(model.non_perpendicular_axes_error)
    if settings.optimize_refraction and num_observations >= 4:
        params_to_optimize.append(model.extra_refraction_coefficient)

    
    optimizer=torch.optim.Adagrad(params_to_optimize)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        time_modelbatch_to_observed_dir = model.dir_given_raw_axis_values(time_modelbatch_to_raw_axis_values)
        
        time_modelbatch_to_loss = compute_dir_losses(time_modelbatch_to_observed_dir, time_modelbatch_to_predicted_dir)
        modelbatch_to_loss = compute_model_losses(time_modelbatch_to_loss, time_dim=0)
        loss = torch.mean(modelbatch_to_loss)
        
        loss.backward()
        optimizer.step()

        if step < 10 or step % int(num_steps / 10) == int(num_steps / 10) - 1 or step == num_steps - 1:
            with torch.no_grad():
                min_loss, min_loss_index = modelbatch_to_loss.min(dim=-1)
                print(f'{debug_string_prefix}step {step + 1} of {num_steps}: avg loss {loss:.3e}, best loss {min_loss:.3e} with model {min_loss_index}')

def align(tracker, alignment_data, targets, settings=AlignmentSettings()):
    num_observations = len(alignment_data)
    num_targets = len(targets)

    print(f"Running alignment with {num_observations} observations and {num_targets} targets...")

    with torch.no_grad():
        times = []
        time_to_raw_axis_values = torch.zeros([num_observations, 2])
        for time_index, alignment_datum in enumerate(alignment_data):
            times.append(alignment_datum.time)
            time_to_raw_axis_values[time_index, :] = torch.tensor(alignment_datum.raw_axis_values)

    print("Looking up target directions...")

    min_z = math.sin(settings.min_alt.to_value(u.rad))
    max_z = math.sin(settings.max_alt.to_value(u.rad))

    with torch.no_grad():
        filtered_targets = []
        target_time_to_predicted_dir = []
        for target in targets:
            time_to_predicted_dir = []
            for time in times:
                altaz = target.altaz_at_time(tracker, time)
                # the coordinate frame seems to be (North, East, Up)
                dir = torch.tensor(altaz.cartesian.xyz.to_value(), dtype=torch.float32)
                dir = dir / torch.linalg.norm(dir, dim=-1, keepdims=True)
                time_to_predicted_dir.append(dir)
            time_to_predicted_dir = torch.stack(time_to_predicted_dir)
            all_too_low = torch.max(time_to_predicted_dir[:, 2]) < min_z
            all_too_high = torch.min(time_to_predicted_dir[:, 2]) > max_z
            if not all_too_high and not all_too_low:
                target_time_to_predicted_dir.append(time_to_predicted_dir)
                filtered_targets.append(target)
        target_time_to_predicted_dir = torch.stack(target_time_to_predicted_dir)

    print(f"Filtered down to {len(filtered_targets)} targets.");

    for index, target in enumerate(filtered_targets):
        print(f'targets[{index}] = {target.display_name}')

    if True:
        print("Running brute alignment...")
        alignment, time_to_target = brute_align(target_time_to_predicted_dir, time_to_raw_axis_values)
    else:
        print("Running random alignment...")
        alignment, time_to_target = random_align(target_time_to_predicted_dir, time_to_raw_axis_values, settings=settings)

    print("Identification:")
    for time_index, filtered_target_index in enumerate(time_to_target):
        target = filtered_targets[filtered_target_index]
        print(f"\tObservation {time_index+1} was {target.display_name}")
        alignment_data[time_index].reconstructed_target = target

    with torch.no_grad():
        time_to_predicted_dir = target_time_to_predicted_dir[time_to_target, torch.arange(num_observations)]

    refine_alignment(alignment, time_to_raw_axis_values, time_to_predicted_dir, num_steps=settings.final_refine_steps, settings=settings, debug_string_prefix="Refining alignment: ")

    print("Accuracy:")
    with torch.no_grad():
        for time_index, filtered_target_index in enumerate(time_to_target):
            target = filtered_targets[filtered_target_index]
            predicted_dir = time_to_predicted_dir[time_index]
            modeled_dir = alignment.dir_given_raw_axis_values(time_to_raw_axis_values[time_index])
            losses = compute_dir_losses(predicted_dir, modeled_dir)
            loss = losses.sum().item()
            misalignment_rad = torch.asin(torch.clamp(torch.linalg.norm(torch.linalg.cross(predicted_dir, modeled_dir), dim=-1), -1.0, 1.0))

            alignment_data[time_index].reconstructed_target = target
            alignment_data[time_index].angular_error = misalignment_rad * u.rad
            alignment_data[time_index].loss = loss
            print(f"\tObservation {time_index}, {target.display_name}, was off by {alignment_data[time_index].angular_error.to(u.deg)}")

    alignment.valid = True

    print(f"Done!\nFinal alignment:\n{alignment}")

    return alignment

def rotation_matrix_around_x(theta):
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    m = torch.stack([
        one,                zero,               zero,
        zero,               torch.cos(theta),   -torch.sin(theta),
        zero,               torch.sin(theta),   torch.cos(theta)
    ], dim=-1).reshape(theta.shape + (3,3))
    return m

def rotation_matrix_around_y(theta):
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    m = torch.stack([
        torch.cos(theta),   zero,               torch.sin(theta),
        zero,               one,                zero,
        -torch.sin(theta),  zero,               torch.cos(theta)
    ], dim=-1).reshape(theta.shape + (3,3))
    return m

def rotation_matrix_around_z(theta):
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)    
    m = torch.stack([
        torch.cos(theta),   -torch.sin(theta),  zero,
        torch.sin(theta),   torch.cos(theta),   zero,
        zero,               zero,               one
    ], dim=-1).reshape(theta.shape + (3,3))
    return m


class AlignmentModel(torch.nn.Module):
    def __init__(self, batch_shape = ()):
        super().__init__()

        # True if we think we've actually aligned something, in which case different tracking modes become possible.
        self.valid = False

        self.batch_shape = batch_shape

        # The az and alt that the encoders would read if the telescope were pointing at 0,0 (due north on horizon).  Thus:
        #   true = raw - offsets
        #   raw = true + offsets
        #   offsets = raw - true
        # axis 0 is azimuth, yaw, or right ascension (the one which needs acceleration when the other is near 90 degrees)
        # axis 1 is altitude, pitch, or declination (which needs no compensation)
        self.encoder_offsets = torch.nn.Parameter(torch.zeros(batch_shape + (2,)))
        self.encoder_offsets.units = u.rad

        self.zenith_roll = torch.nn.Parameter(torch.zeros(batch_shape))
        self.zenith_roll.units = u.rad
        
        self.zenith_pitch = torch.nn.Parameter(torch.zeros(batch_shape))
        self.zenith_pitch.units = u.rad

        # see https://www.wildcard-innovations.com.au/geometric_mount_errors.html
        # for helpful pictures and names.
        self.non_perpendicular_axes_error = torch.nn.Parameter(torch.zeros(batch_shape))
        self.non_perpendicular_axes_error.units = u.rad

        self.collimation_error_in_azimuth = torch.nn.Parameter(torch.zeros(batch_shape))
        self.collimation_error_in_azimuth.units = u.rad
        
        # Whoever gave us the targets should already have included atmospheric refraction using some standard params
        # (TODO: make sure the target sources all match in terms of those params)
        # but the actual refraction may differ.  The simple model Skyfield uses is written in terms of a value depending
        # only on the altitude, times a value depending on the temperature and pressure - this parameter adds onto the latter
        # TODO: when printing, differentiate that this one is not an angle
        self.extra_refraction_coefficient = torch.nn.Parameter(torch.zeros(batch_shape))
        self.extra_refraction_coefficient.units = u.dimensionless_unscaled

    @classmethod
    def choose_submodel(cls, batched_model, index):
        single_model = cls()
        for name, param in single_model.named_parameters():
            with torch.no_grad():
                param[...] = getattr(batched_model, name)[index, ...]
        return single_model
    
    @classmethod
    def choose_submodels(cls, batched_model, indices):
        submodels = cls(batch_shape=indices.shape)
        for name, param in submodels.named_parameters():
            with torch.no_grad():
                param[...] = getattr(batched_model, name)[indices, ...]
        return submodels
    
    def assign_submodel(self, self_index, other, other_index):
        for name, param in self.named_parameters():
            with torch.no_grad():
                param[self_index, ...] = getattr(other, name)[other_index, ...]

    @classmethod
    def concatenate_batches(cls, first_model, second_model):
        sum_batch_shape = (first_model.batch_shape[0] + second_model.batch_shape[0],)
        sum_model = cls(batch_shape=sum_batch_shape)
        for name, param in sum_model.named_parameters():
            with torch.no_grad():
                param[...] = torch.cat((getattr(first_model, name), getattr(second_model, name)), dim=0)
        return sum_model


    def randomize(self):
        self.randomize_encoder_offsets()
        self.randomize_zenith_error()
        self.randomize_mount_errors()

    def randomize_encoder_offsets(self):
        with torch.no_grad():
            self.encoder_offsets[:] = torch.rand(self.batch_shape + (2,)) * 2 * math.pi

    def randomize_zenith_error(self, stdev_radians = math.radians(3.0)):
        with torch.no_grad():
            self.zenith_pitch[...] = torch.randn(self.batch_shape) * stdev_radians
            self.zenith_roll[...] = torch.rand(self.batch_shape) * stdev_radians

    def randomize_mount_errors(self, stdev_radians = math.radians(0.1)):
        with torch.no_grad():
            self.collimation_error_in_azimuth[...] = torch.rand(self.batch_shape) * stdev_radians
            self.non_perpendicular_axes_error[...] = torch.rand(self.batch_shape) * stdev_radians

    def __repr__(self):
        ret = "AlignmentModel\n"
        for name, param in self.named_parameters():
            display_units = param.units
            if param.units.physical_type == u.get_physical_type("angle"):
                display_units = u.deg

            ret += f"\t{name}: {u.Quantity(param.data, unit=param.units).to(display_units)}\n"
        return ret

    def matrix_given_raw_axis_values(self, raw_axis_values):
        # raw_axis_values will be shape
        #   (...) + self.batch_shape + (2,)
        # and we will return
        #   (...) + self.batch_shape + (3, 3)


        

        # How does the telescope move?  Imagine it's sitting on the ground, pointed at the horizon to the north (looking along +x in the North East Down frame).
        # It then undergoes the following rotations:

        # These two are caused by the tripod not being level.  Or, if this were an eqatorial mount rather than AltAz,
        # it would also include the lattitude, such that the zenith would coincide with the celestial pole.
        A = rotation_matrix_around_y(self.zenith_pitch)
        B = rotation_matrix_around_x(self.zenith_roll)

        # This is the azimuth (yaw) of the telescope, including both what the encoders measure, and also the initially-unknown offset.
        C = rotation_matrix_around_z(raw_axis_values[..., 0] - self.encoder_offsets[..., 0])

        # This would be if one of the fork arms were slightly longer than the other, so that when we pitch the telescope, it's not moving straight up.
        D = rotation_matrix_around_x(self.non_perpendicular_axes_error)

        # This is the altitude (pitch) of the telescope, including both what the encoders measure, and also the initially-unknown offset.
        # Note the minus sign, because of the handedness of our coordinate system.
        E = rotation_matrix_around_y(-(raw_axis_values[..., 1] - self.encoder_offsets[..., 1]))

        # This would be if the optical tube were pointing slightly to the right or left, so that when we pitch, it moves in a cone rather than a plane.
        F = rotation_matrix_around_z(self.collimation_error_in_azimuth)        
        
        dir = A @ B @ C @ D @ E @ F

        dir = dir / torch.linalg.norm(dir, dim=-1, keepdims=True)
        return dir

    def dir_given_numpy_raw_axis_values(self, raw_axis_values):
        with torch.no_grad():
            return self.dir_given_raw_axis_values(torch.tensor(raw_axis_values, dtype=torch.float32)).numpy()
        return 

    def dir_given_raw_axis_values(self, raw_axis_values):
        p = torch.tensor([[1.0], [0.0], [0.0]])
        d = self.matrix_given_raw_axis_values(raw_axis_values) @ p
        d = d.squeeze(-1)
        d = self.refract_apparent_to_true(d)
        return d

    def raw_axis_values_given_numpy_dir(self, dir):
        with torch.no_grad():        
            return self.raw_axis_values_given_dir(torch.tensor(dir, dtype=torch.float32)).numpy()

    def raw_axis_values_given_dir(self, dir):
        """
        When going from raw axis values and the rest of our model to a direction, we have:

            r = A B C D E F p

        Where r is the direction we want, p is just a forward vector, and all the matrices ABCDEF
        are dependent on our model's parameters, with particularly C and E dependent on the raw axis values.
        Here, we are given everything except C and E, which we want to solve for.

            (B' A' r) = C D E (F p)

        The quantities in parens are vectors we can compute directly, so we could symbolically expand

        (B'A'r).x       cos(c)  -sin(c) 0       1   0       0           cos(e)  0   sin(e)      (F p).x
        (B'A'r).y   =   sin(c)  cos(c)  0   *   0   cos(d)  -sin(d) *   0       1   0       *   (F p).y
        (B'A'r).z       0       0       1       0   sin(d)  cos(d)      -sin(e) 0   cos(e)      (F p).z

        Which could probably be solved, but I'm thinking of it in a geometric way.  C is a yaw that we
        can choose, D is a roll we are given, and E is a pitch that we can choose.  Together, they must
        bring (F p) into (B' A' R).
        
        Think of it like you're flying a plane.  You have a bug on your windshield at some point off to
        the side of your crosshairs, at point (F p) in the plane's coordinate system.  You'd like to
        point it at a target, which is at point (B' A' r) in the plane's coordinate system.  You can do
        any yaw, but then must roll by non_perpendicular_axes_error, and then can do any pitch.

        Look at the plane from the back.  You must bring the bug up to a horizontal line at z, but that
        line will move as you roll.  The height of the line in the center will be higher: its original
        height over the cosine of the roll angle.  But then it will slope down, at a slope of the
        tangent of the roll angle, as you move sideways to reach the y coordinate of the bug.

        So now you have rolled, we know the z coordinate (in the plane's rolled coordinate frame) you
        must bring the bug up to via pitching.  Consider a view from the side of the (now rolled) plane.
        The bug is not all the way at the front, rather its x is only cos(collimation_error_in_azimuth),
        and we need to bring it to the z we computed (which may be more than that), so we might fail,
        but in any case we can compute it with arccos.

        Then it's a simple matter of inverting some things and subtracting to figure the yaw.
        """

        # todo: support inverting this.  However, this backward pass is only used by random_align, which
        # does not need to search for this value.  We can optimize it with gradient descent later, during
        # refine_alignment().
        assert(self.extra_refraction_coefficient.count_nonzero() == 0)

        dir = dir / torch.norm(dir, dim=-1, keepdim=True)

        # These should match what they are in matrix_given_raw_axis_values()
        A = rotation_matrix_around_y(self.zenith_pitch)
        B = rotation_matrix_around_x(self.zenith_roll)
        D = rotation_matrix_around_x(self.non_perpendicular_axes_error)
        F = rotation_matrix_around_z(self.collimation_error_in_azimuth)
        
        Bt_At_r = (B.mT @ A.mT @ dir.unsqueeze(-1)).squeeze(-1)

        p = torch.tensor([1.0, 0.0, 0.0])
        F_p = (F @ p.unsqueeze(-1)).squeeze(-1)
      
        B_At_r_z = Bt_At_r[..., 2]
        bug_x = torch.cos(self.collimation_error_in_azimuth)
        bug_y = torch.sin(self.collimation_error_in_azimuth)        
        target_z_at_center = B_At_r_z / torch.cos(self.non_perpendicular_axes_error)
        target_z_at_bug = target_z_at_center - bug_y * torch.tan(self.non_perpendicular_axes_error)
        # Note that this could NaN if target_z_at_bug / bug_x is outside (-1, 1), which means you can't get the bug 'high enough' no matter what you do.
        # Let's clamp it, which geomerically means that we'll pitch fully up or down to get as close as possible.
        E_alt = torch.asin(torch.clamp(target_z_at_bug / bug_x, -1.0, 1.0))
        E = rotation_matrix_around_y(-E_alt)  # minus because our handedness is weird
        D_E_F_p = (D @ E @ F_p.unsqueeze(-1)).squeeze(-1)
        D_E_F_p_az = torch.atan2(D_E_F_p[..., 1], D_E_F_p[..., 0])
        Bt_At_r_az = torch.atan2(Bt_At_r[..., 1], Bt_At_r[..., 0])
        C_az = Bt_At_r_az - D_E_F_p_az

        # if we didn't care about the errors, the simpler version of the above would be:
        #   Bt_At_r_azalt = xyz_to_azalt(Bt_At_r)
        #   C_az = Bt_At_r_azalt[..., 0]
        #   E_alt = Bt_At_r_azalt[..., 1]

        raw_axis_values = torch.stack([
            C_az + self.encoder_offsets[..., 0],
            E_alt + self.encoder_offsets[..., 1]
        ], dim=-1)

        if False:
            # check:
            C = rotation_matrix_around_z(C_az)
            C_retcon = rotation_matrix_around_z(raw_axis_values[..., 0] - self.encoder_offsets[..., 0])
            E_retcon = rotation_matrix_around_y(-(raw_axis_values[..., 1] - self.encoder_offsets[..., 1]))
            C_D_E_retcon = C_retcon @ D @ E_retcon
            C_D_E_F_p_retcon = (C_D_E_retcon @ F_p.unsqueeze(-1)).squeeze(-1)

            assert E_retcon.allclose(E, atol=1e-4, equal_nan=True)
            assert C_retcon.allclose(C, atol=1e-5, equal_nan=True)
            assert C_D_E_retcon.allclose(C @ D @ E, atol=1e-5, equal_nan=True)
            assert Bt_At_r.allclose(C_D_E_F_p_retcon, atol=1e-3, equal_nan=True) or torch.isnan(C_D_E_F_p_retcon).any()

        return raw_axis_values
    
    def refract_apparent_to_true(self, apparent_dir):
        apparent_azalt = xyz_to_azalt(apparent_dir)
        apparent_alt = apparent_azalt[..., 1]
        apparent_alt_deg = torch.rad2deg(apparent_alt)

        # These fudge factors are from Skyfield, which apparently come from 
        #   Bennett, G.G. (1982). "The Calculation of Astronomical Refraction in Marine Navigation". Journal of Navigation. 35 (2): 255–259. Bibcode:1982JNav...35..255B. doi:10.1017/S0373463300022037. S2CID 140675736.
        # plus some overly-rounded arcsecond-to-degree conversion (which should bake itself into extra_refraction_coefficient)
        r_deg = 0.016667 / torch.tan(torch.deg2rad((apparent_alt_deg + 7.31 / (apparent_alt_deg + 4.4))))
        d_deg = r_deg * self.extra_refraction_coefficient

        d_deg = torch.where(torch.logical_and(torch.gt(apparent_alt_deg, -1.0), torch.le(apparent_alt_deg, 89.9)), d_deg, 0.0)

        true_alt_deg = apparent_alt_deg - d_deg
        true_alt = torch.deg2rad(true_alt_deg)
        true_azalt = torch.stack([apparent_azalt[..., 0], true_alt], dim=-1)
        true_dir = azalt_to_xyz(true_azalt)
        return true_dir
    
    def raw_axis_values_given_dir_numeric(self, dir):
        raw_axis_values = torch.zeros(2, requires_grad=True)
        optimizer = torch.optim.Adam([raw_axis_values], lr=0.1)
        for _ in range(1000):
            optimizer.zero_grad()
            loss = torch.sum((self.dir_given_raw_axis_values(raw_axis_values) - dir.detach())**2)
            loss.backward()
            optimizer.step()

        print(loss)
        return raw_axis_values.detach()

    

    def forward(self, raw_axis_values):
        return self.dir_given_raw_axis_values(raw_axis_values)


if __name__ == "__main__":
    """
    Testing stuff
    """

    test_model = AlignmentModel()

    #should match:
    #time 2 Arcturus at tensor([-0.2043,  0.5604,  0.8026], dtype=torch.float64)

    arcturus_altaz = torch.tensor([1.92051186, 0.93189984])
    arcturus_dir = torch.tensor([-0.2043,  0.5604,  0.8026])
    reconstructed_arcturus_dir = test_model.dir_given_raw_axis_values(arcturus_altaz)
    print(reconstructed_arcturus_dir)
    print(arcturus_dir)
    assert reconstructed_arcturus_dir.allclose(arcturus_dir, rtol=1e-4, atol=1e-3)

    test_model = AlignmentModel()
    for _ in range(100):
        test_model.randomize()

        raw = test_model.raw_axis_values_given_dir(arcturus_dir)
        reconstructed_arcturus_dir = test_model.dir_given_raw_axis_values(raw)

        print(reconstructed_arcturus_dir)
        print(arcturus_dir)
        assert reconstructed_arcturus_dir.allclose(arcturus_dir, atol=1e-3) or torch.isnan(reconstructed_arcturus_dir).any()

    target_time_to_predicted_dir = torch.zeros(size=(9, 3, 3))
    time_to_raw_axis_values = torch.zeros(size=(3, 2))
    brute_align(target_time_to_predicted_dir, time_to_raw_axis_values)

    print("Tests passed!")


