import math
import time
import astropy.units as u
import astropy.time
import numpy as np
import torch
import torch.nn

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

    def __repr__(self):
        return f'astrolock.model.alignment.AlignmentDatum({repr(self.target)}, {repr(self.time)}, {repr(self.raw_axis_values)})'
    
    def to_json(self):
        return {
            "time": str(self.time),
            "raw_axis_values": self.raw_axis_values.to_value(u.rad).tolist()
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

    - stepper offsets (2 numbers)
    - roll and pitch of azimuth axis (2 numbers, can assume zeros if you leveled/polar aligned it well)
    - axis misalignments (2 numbers, can assume zeros if telescope didn't fall off back of truck)
    - incorrect lattitude / longitude (2 numbers - for distant targets, degenerate with roll and pitch of azimuth axis)
    - incorrect time (1 number - for distant targets, degenerate with roll of azimuth)
    - various fudge factors for mirror flop, 

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

def wrap_to_pi(theta):
    ret = torch.atan2(torch.sin(theta), torch.cos(theta))
    return ret

@torch.no_grad()
def euler_align(target_time_to_predicted_dir, time_to_raw_axis_values):
    """
    A super lame version that assumes we're perfectly horizontal and the axes are perfectly aligned.
    """
    num_targets, num_observations, num_spatial_dimensions = target_time_to_predicted_dir.shape

    target_time_to_predicted_azalt = xyz_to_azalt(target_time_to_predicted_dir)
    
    
    models = []
    # this outer loop isn't strictly necessary and multiplies the amount of work, but makes us
    # resilient to the case where our first observation was further off than the others were.
    for time_to_trust in range(num_observations):
        # for each target, make a model based on the assumption that our trusted observation was of that target.
        for trusted_target in range(num_targets):
            model = AlignmentModel()
            # true = raw - offsets
            # offsets = raw - true
            model.stepper_offsets[:] =  time_to_raw_axis_values[time_to_trust, :] - target_time_to_predicted_azalt[trusted_target, time_to_trust, :]
            models.append(model)

    # figure out where those models pointed at each observation
    batch_size = num_targets
    batch_time_to_observed_dir = torch.zeros((batch_size, num_observations, num_spatial_dimensions))
    for batch_index in range(0, batch_size):            
        batch_time_to_observed_dir[batch_index, :, :] = models[batch_index].dir_given_raw_axis_values(time_to_raw_axis_values[:, :])
    
    batch_target_time_to_dot = torch.einsum('...tod,...od->...to', target_time_to_predicted_dir[:, :, :], batch_time_to_observed_dir)

    (batch_time_to_best_dot, batch_time_to_best_target) = torch.max(batch_target_time_to_dot, dim=-2)

    batch_time_to_lowest_angle = torch.acos(torch.clamp(batch_time_to_best_dot, -1.0, 1.0))

    batch_to_loss = batch_time_to_lowest_angle.square().mean(dim=-1).sqrt()

    (best_loss_in_batch, best_batch_index) = torch.min(batch_to_loss, dim=-1)

    best_loss = best_loss_in_batch
    best_time_to_target = batch_time_to_best_target[best_batch_index, :]
    best_model = models[best_batch_index]

    print(f" loss {best_loss} with assignments {best_time_to_target}")

    return best_model, best_time_to_target


@torch.no_grad()
def random_align(target_time_to_predicted_dir, time_to_raw_axis_values, num_samples = 100000, max_batch_size = 10000):
    """
    Try many random guesses for alignment, and return the best one.  Hopefully, if it's good enough, it will be in the watershed of the global optimum.

    This currently assumes we're level, but it doesn't have to, it could easily sample all parameters of the model.

    It is, however, quite slow.
    """

    num_targets, num_observations, num_spatial_dimensions = target_time_to_predicted_dir.shape
    
    best_loss = math.inf
    best_time_to_target = None
    best_model = None

    for first_sample in range(0, num_samples, max_batch_size):
        batch_size = min(max_batch_size, num_samples - first_sample)

        print(f"Calculating guesses {first_sample + 1}-{first_sample + batch_size} of {num_samples}...", end='')

        models = []
        batch_time_to_observed_dir = torch.zeros((batch_size, num_observations, num_spatial_dimensions))
        for batch_index in range(0, batch_size):
            model = AlignmentModel()
            model.stepper_offsets[:] = torch.rand(2) * (2.0 * math.pi)
            # TODO: could sample other things here, but probably not necessary

            batch_time_to_observed_dir[batch_index, :, :] = model.dir_given_raw_axis_values(time_to_raw_axis_values[:, :])
            models.append(model)

        batch_target_time_to_dot = torch.einsum('...tod,...od->...to', target_time_to_predicted_dir, batch_time_to_observed_dir)

        (batch_time_to_best_dot, batch_time_to_best_target) = torch.max(batch_target_time_to_dot, dim=-2)

        batch_time_to_lowest_angle = torch.acos(torch.clamp(batch_time_to_best_dot, -1.0, 1.0))

        batch_to_loss = batch_time_to_lowest_angle.square().mean(dim=-1).sqrt()

        (best_loss_in_batch, best_batch_index) = torch.min(batch_to_loss, dim=-1)

        if best_loss_in_batch < best_loss:
            best_loss = best_loss_in_batch
            best_time_to_target = batch_time_to_best_target[best_batch_index, :]
            best_model = models[best_batch_index]

        print(f" loss {best_loss} with assignments {best_time_to_target}")

    return best_model, best_time_to_target


def align(tracker, alignment_data, targets, min_alt = -20.0 * u.deg, max_alt = 85.0 * u.deg):
    num_observations = len(alignment_data)
    num_targets = len(targets)

    print(f"Running alignment with {num_observations} observations and {num_targets} targets...")

    times = []
    time_to_raw_axis_values = torch.zeros([num_observations, 2])
    for time_index, alignment_datum in enumerate(alignment_data):
        times.append(alignment_datum.time)
        time_to_raw_axis_values[time_index, :] = torch.tensor(alignment_datum.raw_axis_values.to_value(u.rad))

    print("Looking up target directions...")

    min_z = math.sin(min_alt.to_value(u.rad))
    max_z = math.sin(max_alt.to_value(u.rad))

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
        too_low = torch.max(time_to_predicted_dir[:, 2]) < min_z
        too_high = torch.min(time_to_predicted_dir[:, 2]) > max_z
        if not too_high and not too_low:
            target_time_to_predicted_dir.append(time_to_predicted_dir)
            filtered_targets.append(target)
        else:
            pass
    target_time_to_predicted_dir = torch.stack(target_time_to_predicted_dir)

    print(f"Filtered down to {len(filtered_targets)} targets.");

    #print("Running random alignment...")
    #rough_alignment, time_to_target = random_align(target_time_to_predicted_dir, time_to_raw_axis_values)

    print("Running euler alignment...")
    rough_alignment, time_to_target = euler_align(target_time_to_predicted_dir, time_to_raw_axis_values)

    print("Identification:")
    for time_index, filtered_target_index in enumerate(time_to_target):
        target = filtered_targets[filtered_target_index]
        print(f"\tObservation {time_index} was {target.display_name}")
        alignment_data[time_index].target = target

    # TODO:
    final_alignment = rough_alignment

    print(f"Done!\nFinal alignment:\n{final_alignment}")

    return final_alignment

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
    def __init__(self):
        # see https://www.wildcard-innovations.com.au/geometric_mount_errors.html
        # for helpful pictures and names.

        # axis 0 is azimuth, yaw, or right ascension (the one which needs acceleration when the other is near 90 degrees)
        # axis 1 is altitude, pitch, or declination (which needs no compensation)
        super().__init__()

        # the az and alt that the steppers read when the telescope is pointing at 0,0 (due north on horizon), so:
        #   true = raw - offsets
        #   raw = true + offsets
        #   offsets = raw - true
        self.stepper_offsets = torch.nn.Parameter(torch.tensor([0.0, 0.0]))

        self.azimuth_roll = torch.nn.Parameter(torch.tensor(0.0))
        self.azimuth_pitch = torch.nn.Parameter(torch.tensor(0.0))
        self.non_perpendicular_axes_error = torch.nn.Parameter(torch.tensor(0.0))
        self.collimation_error_in_azimuth = torch.nn.Parameter(torch.tensor(0.0))

        # How does the telescope move?  Imagine it's sitting on the ground, pointed at the horizon to the north (looking along +x in the North East Down frame)
        # - rotate it around local Y by self.azimuth_pitch
        # - rotate it around local X by self.azimuth_roll
        # - rotate it around local Z by self.raw_axis_values[0] - self.stepper_offsets[0]  (azimuth)
        # - rotate it around local X by self.non_perpendicular_axes_error
        # - rotate it around local Y by self.raw_axis_values[1] - self.stepper_offsets[1]  (altitude)
        # - rotate it around local Z by self.collimation_error_in_azimuth

    def __repr__(self):
        ret = "AlignmentModel\n"
        for name, param in self.named_parameters():
            ret += f"\t{name}: {param.data}\n"
        return ret

    def matrix_given_raw_axis_values(self, raw_axis_values):
        A = rotation_matrix_around_y(self.azimuth_pitch)
        B = rotation_matrix_around_x(self.azimuth_roll)
        C = rotation_matrix_around_z(raw_axis_values[..., 0] - self.stepper_offsets[0])     # azimuth
        D = rotation_matrix_around_x(self.non_perpendicular_axes_error)
        E = rotation_matrix_around_y(-(raw_axis_values[..., 1] - self.stepper_offsets[1]))  # altitude
        F = rotation_matrix_around_z(self.collimation_error_in_azimuth)        
        return A @ B @ C @ D @ E @ F
            
    def dir_given_raw_axis_values(self, raw_axis_values):
        p = torch.tensor([[1.0], [0.0], [0.0]])
        d = self.matrix_given_raw_axis_values(raw_axis_values) @ p
        # back to row vectors
        d = d.reshape(d.shape[:-1])
        return d
        
    def raw_axis_values_given_dir_analytic(self, dir):
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

        But that probably gets crazy when D is non-identity, which means we have cross-axis error.
        If instead we assume D is identity, we just have:

            r = A B C E F p
            (B' A' r) = C E (F p)
        
        Which we could also figure symbolically, but really, since we can compute the values in parens,
        we just need to formulate C E in terms of a yaw and pitch so that it solves that equation.

        Think of it like you're flying a plane.  You have a bug on your windshield at some point off to
        the side of your crosshairs, (F p), that you'd like to point in some arbitrary direction (B' A' r),
        using only yaw and pitch (no roll).  It'd normally be possible, but if the bug was very far off to
        the side (like near your wing), and the target point was very high, you might not get the high
        enough no matter how you pitch.  (You'd also need roll...)

        cos(self.collimation_error_in_azimuth) determines the 'arm' of your bug
        lhs[2] is how high we need the bug
        pitch is acos(lhs[2] / cos(self.collimation_error_in_azimuth))   # can be nan if that's > 1

        """

        A = rotation_matrix_around_y(self.azimuth_pitch)
        B = rotation_matrix_around_x(self.azimuth_roll)
        F = rotation_matrix_around_z(self.collimation_error_in_azimuth)
        
        Bt_At_r = (B.mT @ A.mT @ dir.unsqueeze(-1)).squeeze(-1)

        p = torch.tensor([1.0, 0.0, 0.0])
        F_p = (F @ p.unsqueeze(-1)).squeeze(-1)
       
        
        # note that this can NaN if the collimation error is too great, so you can't get the 'bug' high enough with any pitch.
        E_alt = torch.asin(Bt_At_r[..., 2] / torch.cos(self.collimation_error_in_azimuth))
        E = rotation_matrix_around_y(-E_alt)  # minus because our handedness is weird
        E_F_p = (E @ F_p.unsqueeze(-1)).squeeze(-1)
        E_F_p_az = torch.atan2(E_F_p[..., 1], E_F_p[..., 0])
        Bt_At_r_az = torch.atan2(Bt_At_r[..., 1], Bt_At_r[..., 0])
        C_az = Bt_At_r_az - E_F_p_az

        # if we didn't care about collimation_error, the simpler version of the above would be:
        #   Bt_At_r_azalt = xyz_to_azalt(Bt_At_r)
        #   C_az = Bt_At_r_azalt[..., 0]
        #   E_alt = Bt_At_r_azalt[..., 1]

        raw_axis_values = torch.stack([
            C_az + self.stepper_offsets[0],
            E_alt + self.stepper_offsets[1]
        ], dim=-1)

        if True:
            # check:
            C = rotation_matrix_around_z(C_az)
            C_retcon = rotation_matrix_around_z(raw_axis_values[..., 0] - self.stepper_offsets[0])
            E_retcon = rotation_matrix_around_y(-(raw_axis_values[..., 1] - self.stepper_offsets[1]))
            C_E_retcon = C_retcon @ E_retcon
            C_E_F_p_retcon = (C_E_retcon @ F_p.unsqueeze(-1)).squeeze(-1)

            assert E_retcon.allclose(E, atol=1e-5)
            assert C_retcon.allclose(C, atol=1e-5)
            assert C_E_retcon.allclose(C @ E, atol=1e-5)
            assert Bt_At_r.allclose(C_E_F_p_retcon, atol=1e-3)

        return raw_axis_values
    
    def raw_axis_values_given_dir_numeric(self, dir):
        raw_axis_values = torch.zeros(2, requires_grad=True)
        optimizer = torch.optim.Adam([raw_axis_values], lr=0.1)
        for _ in range(1000):
            optimizer.zero_grad()
            loss = torch.sum((self.dir_given_raw_axis_values(raw_axis_values) - dir.detach())**2)
            loss.backward()
            optimizer.step()

        print(loss)
        # Return the optimized raw axis values
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
    #print(reconstructed_arcturus_dir)
    #print(arcturus_dir)
    assert reconstructed_arcturus_dir.allclose(arcturus_dir, atol=1e-3)

    test_model = AlignmentModel()
    for _ in range(100):
        with torch.no_grad():
            test_model.stepper_offsets[:] = torch.rand((2)) * 2 * math.pi
            test_model.azimuth_pitch[...] = torch.randn([]) * 2.0
            test_model.azimuth_roll[...] = torch.rand([]) * 2.0
            
            
            # my confusion is why I can't uncomment this:
            #test_model.collimation_error_in_azimuth[...] = .69 #torch.rand([]) * 0.5
            

            # note that this would fail for analytic, but should work for numeric
            #test_model.non_perpendicular_axes_error[...] = torch.rand([]) * 0.5
            pass

        raw = test_model.raw_axis_values_given_dir_analytic(arcturus_dir)
        reconstructed_arcturus_dir = test_model.dir_given_raw_axis_values(raw)
        print(reconstructed_arcturus_dir)
        print(arcturus_dir)
        assert reconstructed_arcturus_dir.allclose(arcturus_dir, atol=1e-3)

    print("Tests passed!")


