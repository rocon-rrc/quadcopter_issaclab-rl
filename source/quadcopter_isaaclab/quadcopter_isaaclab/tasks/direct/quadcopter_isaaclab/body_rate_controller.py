##in cfg file
# In your_env_config.py
from dataclasses import dataclass

@dataclass
class RateControllerCfg:
    """Configuration for the all-in-one Multicopter Rate Controller."""
    # Roll axis gains
    roll_k: float = 1.0
    roll_p: float = 0.15
    roll_i: float = 0.20
    roll_d: float = 0.003
    roll_ff: float = 0.0
    roll_int_lim: float = 0.30
    
    # Pitch axis gains
    pitch_k: float = 1.0
    pitch_p: float = 0.15
    pitch_i: float = 0.20
    pitch_d: float = 0.003
    pitch_ff: float = 0.0
    pitch_int_lim: float = 0.30
    
    # Yaw axis gains
    yaw_k: float = 1.0
    yaw_p: float = 0.2
    yaw_i: float = 0.1
    yaw_d: float = 0.0
    yaw_ff: float = 0.0
    yaw_int_lim: float = 0.30
    
    # Yaw torque filter cutoff frequency in Hz
    yaw_tq_cutoff: float = 5.0

##in the top part of env file

# You can place this class at the top of your environment's Python file
class MulticopterRateController:
    """
    An all-in-one, vectorized, PyTorch-based rate controller that mimics the PX4 implementation.
    Designed for high-performance use within Isaac Lab.
    """
    def __init__(self, cfg: RateControllerCfg, num_envs: int, dt: float, device: str):
        """
        Initializes the rate controller.

        Args:
            cfg (RateControllerCfg): The configuration dataclass object.
            num_envs (int): The number of parallel environments.
            dt (float): The simulation time step.
            device (str): The PyTorch device (e.g., 'cuda:0' or 'cpu').
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.dt = dt
        self.device = device

        # --- State Variables ---
        # PID integral term for each axis [roll, pitch, yaw]
        self._integral = torch.zeros(self.num_envs, 3, device=self.device)
        # Low-pass filter output state for the yaw torque
        self._lpf_yaw_output = torch.zeros(self.num_envs, device=self.device)

        # --- Parameters (loaded from config) ---
        # We store gains as (3,) tensors for easy broadcasting
        self._gains_p = torch.zeros(3, device=self.device)
        self._gains_i = torch.zeros(3, device=self.device)
        self._gains_d = torch.zeros(3, device=self.device)
        self._gains_ff = torch.zeros(3, device=self.device)
        self._integrator_limit = torch.zeros(3, device=self.device)
        
        # Load parameters from the config object
        self.update_parameters()

    def update_parameters(self):
        """Loads and computes the PID gains from the configuration object."""
        rate_k = torch.tensor([self.cfg.roll_k, self.cfg.pitch_k, self.cfg.yaw_k], device=self.device)
        
        self._gains_p = rate_k * torch.tensor([self.cfg.roll_p, self.cfg.pitch_p, self.cfg.yaw_p], device=self.device)
        self._gains_i = rate_k * torch.tensor([self.cfg.roll_i, self.cfg.pitch_i, self.cfg.yaw_i], device=self.device)
        self._gains_d = rate_k * torch.tensor([self.cfg.roll_d, self.cfg.pitch_d, self.cfg.yaw_d], device=self.device)
        self._gains_ff = torch.tensor([self.cfg.roll_ff, self.cfg.pitch_ff, self.cfg.yaw_ff], device=self.device)
        self._integrator_limit = torch.tensor([self.cfg.roll_int_lim, self.cfg.pitch_int_lim, self.cfg.yaw_int_lim], device=self.device)

    def run(self, rate_setpoints: torch.Tensor, current_rates: torch.Tensor, angular_accel: torch.Tensor) -> torch.Tensor:
        """
        Runs one step of the rate control loop for all environments.

        Args:
            rate_setpoints (torch.Tensor): Desired angular rates [roll, pitch, yaw] in rad/s. Shape: (num_envs, 3).
            current_rates (torch.Tensor): Current angular rates from the simulator. Shape: (num_envs, 3).
            angular_accel (torch.Tensor): Current angular acceleration. Shape: (num_envs, 3).

        Returns:
            torch.Tensor: The final torque command [x, y, z]. Shape: (num_envs, 3).
        """
        # --- PID Calculation ---
        rate_error = rate_setpoints - current_rates

        # Update and clamp the integral term (anti-windup)
        self._integral += rate_error * self.dt
        self._integral = torch.clip(self._integral, -self._integrator_limit, self._integrator_limit)
        
        # Calculate PID terms using vectorized operations (broadcasting gains)
        p_term = self._gains_p * rate_error
        i_term = self._gains_i * self._integral
        d_term = self._gains_d * -angular_accel
        ff_term = self._gains_ff * rate_setpoints
        
        torque_setpoint = p_term + i_term + d_term + ff_term

        # --- Yaw Torque Low-Pass Filter ---
        yaw_torque_in = torque_setpoint[:, 2]
        if self.cfg.yaw_tq_cutoff > 0.0:
            rc = 1.0 / (2.0 * torch.pi * self.cfg.yaw_tq_cutoff)
            alpha = self.dt / (rc + self.dt)
            self._lpf_yaw_output = alpha * yaw_torque_in + (1.0 - alpha) * self._lpf_yaw_output
        else:
            self._lpf_yaw_output = yaw_torque_in # Filter is disabled
        
        torque_setpoint[:, 2] = self._lpf_yaw_output
        
        return torque_setpoint

    def reset_idx(self, env_ids: torch.Tensor):
        """Resets the controller's internal state for specified environments."""
        if len(env_ids) == 0:
            return
        
        self._integral[env_ids] = 0.0
        self._lpf_yaw_output[env_ids] = 0.0

##in the main part of env file
# In your main environment file (e.g., your_env.py)

# from your_env_config import RateControllerCfg # Make sure to import the config class
# (The MulticopterRateController class definition from above goes here)

class YourDroneEnv(RLTask):
    def __init__(self, cfg, ...):
        # Instantiate the config from the dictionary passed by Hydra
        self.cfg = hydra.utils.instantiate(cfg) 
        
        # ... your other __init__ setup (num_envs, device, etc.) ...
        
        # Create the rate controller instance
        self.rate_controller = MulticopterRateController(
            cfg=self.cfg.task.env.rate_controller, # Pass the config object from your main config
            num_envs=self.num_envs,
            dt=self.sim_params.dt,
            device=self.device
        )
        
        # Buffer to store previous rates for calculating angular acceleration
        self.previous_body_rates = torch.zeros_like(self.rigid_body_states[:, 0, 7:10])

    def reset_idx(self, env_ids):
        # ... your existing reset logic (resetting root states, etc.)
        
        # Reset the controller's state for the environments being reset
        self.rate_controller.reset_idx(env_ids)
        
        # Also reset the previous rates buffer
        self.previous_body_rates[env_ids] = 0.0

    def pre_physics_step(self, actions):
        # Get actions from the policy
        # actions shape: (num_envs, 4) -> [thrust, roll_rate, pitch_rate, yaw_rate]
        
        # 1. Interpret NN actions
        # Scale rate setpoints from [-1, 1] to a physical range (e.g., +/- 400 deg/s)
        max_rates = torch.tensor([400.0, 400.0, 200.0], device=self.device) * (torch.pi / 180.0)
        rate_setpoints = actions[:, 1:4] * max_rates
        
        # 2. Get current state from simulation
        current_body_rates = self.rigid_body_states[:, self.drone_indices, 7:10].squeeze(1)
        
        # Calculate angular acceleration using finite difference
        angular_accel = (current_body_rates - self.previous_body_rates) / self.dt
        
        # 3. Run the controller to get torques
        torques = self.rate_controller.run(
            rate_setpoints,
            current_body_rates,
            angular_accel
        )
        
        # 4. Calculate thrust force
        thrust_cmd = actions[:, 0]
        max_thrust_force = 15.0 # Newtons, should also be in your config!
        forces = torch.zeros((self.num_envs, 3), device=self.device)
        forces[:, 2] = (thrust_cmd + 1.0) / 2.0 * max_thrust_force # Scale [-1, 1] to [0, MAX]
        
        # 5. Apply forces and torques to the simulation
        self.rigid_body_forces[:, self.drone_indices, :] = forces.unsqueeze(1)
        self.rigid_body_torques[:, self.drone_indices, :] = torques.unsqueeze(1)

        # 6. Store current rates for the next step's calculation
        self.previous_body_rates[:] = current_body_rates



#do it in post physics instead, in order to inc pid freq:
# In your Isaac Lab Environment Task class (e.g., YourDroneEnv)

class YourDroneEnv(RLTask):
    def __init__(self, cfg, ...):
        # ... (Same setup as before) ...
        self.rate_controller = MulticopterRateController(...)
        self.previous_body_rates = torch.zeros(...)
        
        # Buffer to hold the action from the policy for the sub-steps
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

    # pre_physics_step now only receives the actions and stores them
    def pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    # post_physics_step is where you will now run the controller
    # This method is called AFTER every single physics step (i.e., at 100 Hz)
    def post_physics_step(self):
        # This is the new home for your controller logic
        
        # 1. Interpret NN actions (these are held constant for `decimation` steps)
        max_rates = torch.tensor([400.0, 400.0, 200.0], device=self.device) * (torch.pi / 180.0)
        rate_setpoints = self.actions[:, 1:4] * max_rates
        
        # 2. Get current state from simulation (fresh data at 100 Hz)
        current_body_rates = self.rigid_body_states[:, self.drone_indices, 7:10].squeeze(1)
        angular_accel = (current_body_rates - self.previous_body_rates) / self.dt
        
        # 3. Run the controller to get torques at 100 Hz
        torques = self.rate_controller.run(
            rate_setpoints,
            current_body_rates,
            angular_accel
        )
        
        # 4. Calculate thrust force (also held constant)
        thrust_cmd = self.actions[:, 0]
        max_thrust_force = 15.0 
        forces = torch.zeros((self.num_envs, 3), device=self.device)
        forces[:, 2] = (thrust_cmd + 1.0) / 2.0 * max_thrust_force
        
        # 5. Apply forces and torques for the NEXT physics step
        self.rigid_body_forces[:, self.drone_indices, :] = forces.unsqueeze(1)
        self.rigid_body_torques[:, self.drone_indices, :] = torques.unsqueeze(1)

        # 6. Store current rates for the next step's calculation
        self.previous_body_rates[:] = current_body_rates
        
        # 7. Your other logic (compute observations, rewards, resets) will
        #    still be wrapped in an `if self.progress_buf[0] % self.decimation == 0:`
        #    block or handled by the parent RLTask class.
        
    def reset_idx(self, env_ids):
        # ... (This logic remains the same)
        self.rate_controller.reset_idx(env_ids)
        self.previous_body_rates[env_ids] = 0.0
        self.actions[env_ids] = 0.0