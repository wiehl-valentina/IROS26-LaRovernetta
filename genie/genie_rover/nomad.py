"""Optional NoMaD proposal backend, ROS-free and with lazy ML imports.

Inference protocol follows visualnav-transformer deployment/src/explore.py.
RGB -> goal-masked diffusion -> unnormalized deltas -> cumulative waypoints.
The model NEVER receives GPS or sends commands. The existing BEV planner ranks
its proposals. External packages and trusted weights must be installed explicitly.
"""
from collections import deque
from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
import math
import numpy as np


@dataclass
class NoMaDConfig:
    model_config_path: str = 'third_party/visualnav-transformer/train/config/nomad.yaml'
    checkpoint_path: str = 'checkpoints/nomad.pth'
    device: str = 'auto'
    num_samples: int = 16
    waypoint_scale_m: float = 0.10
    scale_calibrated: bool = False
    goal_weight: float = 1.0
    min_context_interval_s: float = 0.2
    max_context_gap_s: float = 5.0
    max_observation_age_s: float = 2.0
    seed: int = 42
    action_min: tuple = (-2.5, -4.0)
    action_max: tuple = (5.0, 4.0)

    def __post_init__(self):
        if isinstance(self.num_samples, bool) or not isinstance(self.num_samples, Integral) or self.num_samples < 1:
            raise ValueError('nomad.num_samples must be a positive integer')
        for name in ('waypoint_scale_m', 'min_context_interval_s', 'max_context_gap_s', 'max_observation_age_s'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'nomad.{name} must be finite and positive')
        if self.max_context_gap_s <= self.min_context_interval_s:
            raise ValueError('nomad.max_context_gap_s must exceed min_context_interval_s')
        if not math.isfinite(self.goal_weight) or self.goal_weight < 0:
            raise ValueError('nomad.goal_weight must be finite and nonnegative')
        lo, hi = np.asarray(self.action_min), np.asarray(self.action_max)
        if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo, hi]).all() or not (hi > lo).all():
            raise ValueError('nomad.action_min/max must be finite ordered 2D bounds')


def decode_actions(normalized_deltas, config):
    """Return [right, forward] metres, including the rover origin at t=0."""
    a = np.asarray(normalized_deltas, dtype=np.float32)
    if a.ndim != 3 or a.shape[-1] != 2 or a.shape[1] < 2:
        raise ValueError('NoMaD output must have shape [samples, horizon, 2]')
    lo, hi = np.asarray(config.action_min), np.asarray(config.action_max)
    deltas = (a+1)*0.5*(hi-lo)+lo
    forward_left = np.cumsum(deltas, axis=1)*config.waypoint_scale_m
    right_forward = np.stack((-forward_left[..., 1], forward_left[..., 0]), axis=-1)
    return np.concatenate((np.zeros((len(a), 1, 2)), right_forward), axis=1)


class NoMaDRunner:
    def __init__(self, config):
        import yaml
        root = Path(__file__).resolve().parents[1]
        config_path = root / config.model_config_path
        checkpoint = root / config.checkpoint_path
        for path in (config_path, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f'NoMaD: falta {path}. Ver genie/README_NOMAD.md')
        params = yaml.safe_load(config_path.read_text())
        if params.get('model_type') != 'nomad' or params.get('vision_encoder') != 'nomad_vint':
            raise ValueError('Supported checkpoint architecture: nomad / nomad_vint')
        if params.get('normalize') is not True or params.get('learn_angle', False):
            raise ValueError('NoMaD requires normalized XY actions without learned angle')
        self.context_length = int(params['context_size'])+1
        self.horizon = int(params['len_traj_pred'])
        if self.context_length < 1 or self.horizon < 4 or self.horizon % 4:
            raise ValueError('Invalid NoMaD context/horizon (horizon must be a multiple of 4)')
        self.image_size = tuple(params['image_size'])
        self.steps = int(params['num_diffusion_iters'])
        if len(self.image_size) != 2 or min(self.image_size) < 1 or self.steps < 1:
            raise ValueError('Invalid NoMaD image size / diffusion steps')
        try:
            import torch
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
            from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
            from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
            from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
        except ImportError as exc:
            raise ImportError('NoMaD optional dependencies missing. See genie/README_NOMAD.md') from exc
        self.torch = torch
        self.config = config
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                                   if config.device == 'auto' else config.device)
        encoder = NoMaD_ViNT(
            obs_encoding_size=params['encoding_size'], context_size=params['context_size'],
            mha_num_attention_heads=params['mha_num_attention_heads'],
            mha_num_attention_layers=params['mha_num_attention_layers'],
            mha_ff_dim_factor=params['mha_ff_dim_factor'])
        encoder = replace_bn_with_gn(encoder)
        decoder = ConditionalUnet1D(input_dim=2, global_cond_dim=params['encoding_size'],
                                   down_dims=params['down_dims'], cond_predict_scale=params['cond_predict_scale'])
        self.model = NoMaD(vision_encoder=encoder, noise_pred_net=decoder,
                           dist_pred_net=DenseNetwork(embedding_dim=params['encoding_size']))
        # Official nomad.pth is a state_dict. No arbitrary pickle objects or
        # strict=False: wrong checkpoints must fail rather than move randomly.
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()
        self.scheduler = DDPMScheduler(num_train_timesteps=self.steps,
                                       beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True, prediction_type='epsilon')
        self.generator = torch.Generator(device=self.device).manual_seed(config.seed)

    def sample(self, images):
        from PIL import Image
        t = self.torch
        if len(images) != self.context_length:
            raise ValueError('NoMaD: incomplete temporal context')
        # Match official RGB resize (no center crop) and ImageNet normalization.
        mean = np.array([.485, .456, .406], dtype=np.float32)
        std = np.array([.229, .224, .225], dtype=np.float32)
        frames = []
        for rgb in images:
            resized = np.asarray(Image.fromarray(rgb).resize(self.image_size), dtype=np.float32)/255
            frames.append(((resized-mean)/std).transpose(2, 0, 1))
        observation = t.from_numpy(np.concatenate(frames, axis=0)[None]).to(self.device)
        with t.inference_mode():
            # Masked goal is irrelevant to policy; using zeros avoids extra RNG.
            goal = t.zeros_like(observation[:, :3])
            mask = t.ones(1, dtype=t.long, device=self.device)
            cond = self.model('vision_encoder', obs_img=observation, goal_img=goal, input_goal_mask=mask)
            cond = cond.repeat(self.config.num_samples, *([1]*(cond.ndim-1)))
            actions = t.randn((self.config.num_samples, self.horizon, 2),
                              device=self.device, generator=self.generator)
            self.scheduler.set_timesteps(self.steps)
            for step in self.scheduler.timesteps:
                noise = self.model('noise_pred_net', sample=actions, timestep=step, global_cond=cond)
                actions = self.scheduler.step(noise, step, actions, generator=self.generator).prev_sample
        return decode_actions(actions.cpu().numpy(), self.config)


def candidate_bank(paths, bev_shape, resolution, forward_range, grid_size, num_points):
    """Metric XY proposals -> uniformly sampled planner row/col curves.

    Reject NaN, reverse, stationary and out-of-window proposals. Do not clip
    endpoints or invent substitute polynomials. PathFollower is forward-only.
    """
    h, w = bev_shape
    bank = []
    for p in np.asarray(paths):
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            continue
        if np.linalg.norm(p[0]) > 1e-5 or (np.diff(p[:, 1]) < -1e-5).any():
            continue
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        if arc[-1] < .05 or p[-1, 1] < .02:
            continue
        keep = np.r_[True, np.diff(arc) > 1e-8]
        distance = np.linspace(0, arc[-1], num_points)
        xy = np.column_stack([np.interp(distance, arc[keep], p[keep, i]) for i in range(2)])
        rows = (h-1)*(1-xy[:, 1]/forward_range)
        cols = w//2+xy[:, 0]/resolution
        if (rows < 0).any() or (rows > h-1).any() or (cols < 0).any() or (cols > w-1).any():
            continue
        bank.append(np.column_stack((rows*(grid_size-1)/(h-1),
                                     cols*(grid_size-1)/(w-1))).astype(np.float32))
    return bank


class NoMaDPlanner:
    def __init__(self, config, dry_run=True, runner=None):
        self.config = config
        if not dry_run and config.scale_calibrated is not True:
            raise ValueError('NoMaD real requiere nomad.scale_calibrated: true y escala medida')
        self.runner = runner if runner is not None else NoMaDRunner(config)
        self.frames = deque(maxlen=self.runner.context_length)
        self.last_timestamp = None
        self.last_added_at = None
        self.new_frame = False

    def observe(self, rgb, timestamp, now):
        self.new_frame = False
        if not np.isfinite(timestamp):
            raise ValueError('NoMaD: invalid frame timestamp')
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            if timestamp < self.last_timestamp:
                self.frames.clear()
                self.last_added_at = None
            else:
                return
        self.last_timestamp = timestamp
        if self.last_added_at is not None:
            gap = now-self.last_added_at
            if gap > self.config.max_context_gap_s or gap < 0:
                self.frames.clear()
            elif gap < self.config.min_context_interval_s:
                return
        self.frames.append(np.array(rgb, dtype=np.uint8, copy=True))
        self.last_added_at = now
        self.new_frame = True

    @property
    def ready(self):
        return self.new_frame and len(self.frames) == self.runner.context_length

    def plan(self, bev, observed, goal_x, goal_y, resolution, forward_range, planner_config):
        from genie_path_planner.planner import plan_on_bev
        if not self.ready:
            raise RuntimeError('NoMaD temporal context is not ready')
        paths = self.runner.sample(list(self.frames))
        bank = candidate_bank(paths, bev.shape, resolution, forward_range,
                              planner_config.grid_size, planner_config.path_num_samples+1)
        # Evaluate whole trajectories and choose ONE actual proposal. A blend
        # of modes could cross an obstacle even if both originals are valid.
        cfg = replace(planner_config, best_k=1,
                      number_of_points_to_filter=planner_config.path_num_samples+1)
        plan = plan_on_bev(bev, observed, goal_x, goal_y, resolution, cfg,
                           candidate_path_bank=bank, bev_forward_range_m=forward_range,
                           candidate_goal_weight=self.config.goal_weight, select_best_candidate=True)
        plan.metadata.update(generator='nomad', raw_candidates=len(paths), metric_candidates=len(bank))
        return plan
