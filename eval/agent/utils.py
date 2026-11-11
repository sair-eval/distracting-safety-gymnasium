"""Observation wrappers used in training and evaluation: front + back camera frame stacking and action repeat."""
from collections import deque

import cv2
import gym
import numpy as np

OBS_REPR_CHANNELS = {'rgb': 3, 'gray_edge': 2, 'rgb_edge': 4}


def apply_obs_repr(rgb, mode):
    """Deterministic observation representation, applied identically at train and eval time.

    This is preprocessing, not augmentation: no extra renderings, no privileged simulator access,
    nothing stochastic, and it applies unchanged to any pixel method. Motivation, measured on the
    paired clean/distracted set: recolouring moves raw RGB 2.35x more than a genuine state change,
    but only 0.26x in thresholded contours. Colour is dominated by the distractor while shape
    survives it, and under per-episode geom recolouring colour carries no object identity anyway.

      'rgb'       identity, 3 channels (default, unchanged behaviour)
      'gray_edge' luminance + Sobel edge magnitude, 2 channels (drops colour entirely)
      'rgb_edge'  RGB + edge magnitude, 4 channels (keeps colour, adds shape)
    """
    if mode == 'rgb':
        return rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(gx * gx + gy * gy), 0, 255).astype(np.uint8)
    if mode == 'gray_edge':
        return np.stack([gray, edge], axis=-1)
    if mode == 'rgb_edge':
        return np.concatenate([rgb, edge[..., None]], axis=-1)
    raise ValueError(f'unknown obs_repr {mode!r}')


class FrameStack(gym.Wrapper):
    """Frame stacking and resizing in one function."""
    def __init__(self, env, k, resize_shape=(64, 64), vision_key='vision', obs_repr='rgb',
                 frame_offsets=None):
        super().__init__(env)
        self._k = k
        # Which past frames make up the stack, newest-first, in agent steps. Default
        # (0, 1, ..., k-1) is the usual consecutive stack. A wider pattern such as (0, 1, 8)
        # keeps the consecutive pair that carries velocity -- the only velocity cue available
        # from pure pixels -- while reaching further back for context. Measured motivation on
        # SafetyPointButton1: the goal marker is absent from BOTH cameras in 44% of states, and
        # three consecutive frames raise coverage only 52% -> 54.7% because they are nearly
        # identical.
        self._offsets = tuple(range(k)) if frame_offsets is None else tuple(frame_offsets)
        assert len(self._offsets) == k, f'need {k} offsets, got {self._offsets}'
        self._frames = deque([], maxlen=max(self._offsets) + 1)
        self.resize_shape = resize_shape
        self.vision_key = vision_key
        self.obs_repr = obs_repr
        per_view = OBS_REPR_CHANNELS[obs_repr]
        
        # Get the original observation shape using the configured key
        if hasattr(env, 'observation_space') and isinstance(env.observation_space, gym.spaces.Dict):
            if self.vision_key == 'vision_front_back':
                # Two views concatenated along channels
                channels = 2 * per_view
                original_shape = (resize_shape[0], resize_shape[1], channels)
            else:
                key = self.vision_key if self.vision_key in env.observation_space.spaces else 'vision'
                original_shape = env.observation_space[key].shape
                channels = per_view
        else:
            original_shape = env.observation_space.shape
            # If flat vector, we will source RGB via render(); force channels=3 or 6 for front_back
            channels = (2 * per_view) if self.vision_key == 'vision_front_back' else per_view
        print("original shape: ", original_shape)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(resize_shape[0], resize_shape[1], channels * k),
            dtype=np.uint8
        )

    def reset(self):
        reset_obs = self.env.reset()
        obs = self._get_vision_obs(reset_obs)

        # Initialize the frame stack with the first observation
        for _ in range(self._frames.maxlen):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._frames.append(self._get_vision_obs(obs))
        return self._get_obs(), reward, cost, done, info

    def _get_obs(self):
        assert len(self._frames) == self._frames.maxlen
        # self._frames is oldest-first; offset 0 is the newest frame.
        sel = [self._frames[-1 - o] for o in self._offsets]
        return np.concatenate(sel[::-1], axis=-1)   # oldest-first, as before

    def _get_vision_obs(self, obs):
        # Handle tuple observations
        if isinstance(obs, tuple) and len(obs) == 2:
            obs, _ = obs

        def ensure_hwc_rgb(img):
            if img is None:
                return None
            if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
                img = np.transpose(img, (1, 2, 0))
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            return img

        if self.vision_key == 'vision_front_back':
            front = None
            back = None

            # Try to use dict keys if present
            if isinstance(obs, dict):
                if 'image' in obs and 'image2' in obs:
                    front = obs['image']
                    back = obs['image2']
                else:
                    # Common alternatives
                    if 'vision' in obs:
                        front = obs['vision']
                    if 'vision_back' in obs:
                        back = obs['vision_back']

            # Fallback: render from env task using camera names
            if (front is None or back is None) and hasattr(self.env, 'task') and hasattr(self.env.task, 'render'):
                try:
                    if front is None:
                        front = self.env.task.render(width=self.resize_shape[0], height=self.resize_shape[1], mode='rgb_array', camera_name='vision', cost={})
                    if back is None:
                        back = self.env.task.render(width=self.resize_shape[0], height=self.resize_shape[1], mode='rgb_array', camera_name='vision_back', cost={})
                except Exception:
                    pass

            # Fallback to generic render once, duplicate if back missing
            if front is None and hasattr(self.env, 'render'):
                try:
                    front = self.env.render()
                except Exception:
                    front = None
            if back is None:
                back = front

            front = ensure_hwc_rgb(front)
            back = ensure_hwc_rgb(back)
            if front is None:
                front = np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8)
            if back is None:
                back = np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8)

            front = cv2.resize(front, self.resize_shape, interpolation=cv2.INTER_AREA)
            back = cv2.resize(back, self.resize_shape, interpolation=cv2.INTER_AREA)

            # Concatenate along channels: 3+3 → 6
            vision6 = np.concatenate([apply_obs_repr(front, self.obs_repr),
                                      apply_obs_repr(back, self.obs_repr)], axis=-1)
            return vision6

        # Single-view path
        vision = None

        # Prefer dict-provided vision if available
        if isinstance(obs, dict):
            key = self.vision_key if self.vision_key in obs else ('vision' if 'vision' in obs else None)
            if key is not None:
                vision = obs[key]

        # Fallback: use rendered frame from environment (RGB)
        if vision is None and hasattr(self.env, 'render'):
            try:
                frame = self.env.render()
                if frame is not None:
                    vision = frame
            except Exception:
                vision = None

        # Last resort: create a black RGB frame to keep pipeline running
        if vision is None:
            vision = np.zeros((self.resize_shape[0], self.resize_shape[1], 3), dtype=np.uint8)

        vision = ensure_hwc_rgb(vision)
        # Resize the vision observation
        resized_vision = cv2.resize(vision, self.resize_shape, interpolation=cv2.INTER_AREA)
        return apply_obs_repr(resized_vision, self.obs_repr)

class ActionRepeat:
    """Action repeat wrapper for safety gymnasium environments."""
    
    def __init__(self, env, action_repeat):
        self.env = env
        self.action_repeat = action_repeat
        self._last_action = None
        self._step_count = 0
        self._episode_steps = 0
        
        # Validate action_repeat
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be >= 1, got {action_repeat}")
        
        print(f"ActionRepeat wrapper initialized with action_repeat={action_repeat}")
        
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped environment."""
        return getattr(self.env, name)
    
    def reset(self, **kwargs):
        """Reset the environment and action repeat state."""
        self._last_action = None
        self._step_count = 0
        self._episode_steps = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """Step the environment with action repeat."""
        self._last_action = action
        self._step_count = 0
        
        total_reward = 0
        total_cost = 0
        done = False
        info = {}
        obs = None
        
        # Repeat the action for action_repeat steps
        for i in range(self.action_repeat):
            if done:
                break
                
            obs, reward, cost, done, info = self.env.step(action)
            total_reward += reward
            total_cost += cost
            self._step_count += 1
            self._episode_steps += 1
            
        # Add action repeat info to the info dict
        if info is None:
            info = {}
        info['action_repeat_steps'] = self._step_count
        info['episode_steps'] = self._episode_steps
            
        return obs, total_reward, total_cost, done, info
    
    @property
    def action_space(self):
        """Return the action space of the wrapped environment."""
        return self.env.action_space
    
    @property
    def observation_space(self):
        """Return the observation space of the wrapped environment."""
        return self.env.observation_space
    
    def render(self, mode='rgb_array'):
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        return self.env.close()
    
    def seed(self, seed=None):
        """Set the random seed."""
        return self.env.seed(seed)
    
    @property
    def unwrapped(self):
        """Return the unwrapped environment."""
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
    
    def get_action_repeat(self):
        """Get the action repeat value."""
        return self.action_repeat
    
    def get_step_count(self):
        """Get the current step count for the last action."""
        return self._step_count
    
    def get_episode_steps(self):
        """Get the total episode steps."""
        return self._episode_steps
