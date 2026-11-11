# This code is based on the code from The Distracting Control Suite: https://github.com/google-research/google-research/tree/master/distracting_control
# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Video background distractions for SafetyGym environments."""
import os
import collections
import gc
from PIL import Image
import numpy as np
import mujoco

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Video backgrounds will use basic image processing.")

DAVIS17_TRAINING_VIDEOS = [
    'bear', 'bmx-bumps', 'boat', 'boxing-fisheye', 'breakdance-flare', 'bus',
    'car-turn', 'cat-girl', 'classic-car', 'color-run', 'crossing',
    'dance-jump', 'dancing', 'disc-jockey', 'dog-agility', 'dog-gooses',
    'dogs-scale', 'drift-turn', 'drone', 'elephant', 'flamingo', 'hike',
    'hockey', 'horsejump-low', 'kid-football', 'kite-walk', 'koala',
    'lady-running', 'lindy-hop', 'longboard', 'lucia', 'mallard-fly',
    'mallard-water', 'miami-surf', 'motocross-bumps', 'motorbike', 'night-race',
    'paragliding', 'planes-water', 'rallye', 'rhino', 'rollerblade',
    'schoolgirls', 'scooter-board', 'scooter-gray', 'sheep', 'skate-park',
    'snowboard', 'soccerball', 'stroller', 'stunt', 'surf', 'swing', 'tennis',
    'tractor-sand', 'train', 'tuk-tuk', 'upside-down', 'varanus-cage', 'walking'
]
DAVIS17_VALIDATION_VIDEOS = [
    'bike-packing', 'blackswan', 'bmx-trees', 'breakdance', 'camel',
    'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'dogs-jump',
    'drift-chicane', 'drift-straight', 'goat', 'gold-fish', 'horsejump-high',
    'india', 'judo', 'kite-surf', 'lab-coat', 'libby', 'loading', 'mbike-trick',
    'motocross-jump', 'paragliding-launch', 'parkour', 'pigs', 'scooter-black',
    'shooting', 'soapbox'
]

# SafetyGym specific constants
SKY_TEXTURE_NAME = 'skybox'  # Name of skybox texture in SafetyGym
Texture = collections.namedtuple('Texture', ('size', 'address', 'textures'))


def imread(filename):
    """Load image from file."""
    img = Image.open(filename)
    img_np = np.asarray(img)
    return img_np


def size_and_flatten(image, ref_height, ref_width):
    """Resize image if necessary and flatten the result."""
    image_height, image_width = image.shape[:2]
    
    if image_height != ref_height or image_width != ref_width:
        if TF_AVAILABLE:
            image = tf.cast(tf.image.resize(image, [ref_height, ref_width]), tf.uint8)
            return tf.reshape(image, [-1]).numpy()
        else:
            # Fallback to PIL for resizing
            img = Image.fromarray(image)
            img = img.resize((ref_width, ref_height))
            image = np.asarray(img)
            return image.flatten()
    return image.flatten()


def create_skybox_cubemap(image, face_size):
    """Create a 6-face cubemap from a single image for MuJoCo skybox.
    
    MuJoCo skybox textures are stored as a cubemap with 6 faces arranged in a specific layout.
    This function creates 6 identical faces from a single image and arranges them properly.
    
    For MuJoCo skyboxes defined with separate face files (@fileright, @fileleft, etc.),
    the faces are typically arranged in memory as a vertical strip in the order:
    [right, left, back, front, down, up]
    
    Args:
        image: Input image (H, W, C) to replicate on all 6 faces
        face_size: Size of each face (assumes square faces)
        
    Returns:
        Flattened cubemap texture data ready for MuJoCo tex_rgb
    """
    # Ensure image is RGB
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]  # Remove alpha channel
    elif len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
    
    # Resize image to face size
    if TF_AVAILABLE:
        face_image = tf.cast(tf.image.resize(image, [face_size, face_size]), tf.uint8).numpy()
    else:
        img = Image.fromarray(image)
        img = img.resize((face_size, face_size))
        face_image = np.asarray(img)
    
    # MuJoCo cubemap faces order (based on assets.yaml):
    # @fileright, @fileleft, @fileback, @filefront, @filedown, @fileup
    # This corresponds to: +X, -X, +Z, -Z, -Y, +Y in typical cubemap notation
    cubemap_height = face_size * 6  # 6 faces stacked vertically
    cubemap_width = face_size
    cubemap = np.zeros((cubemap_height, cubemap_width, 3), dtype=np.uint8)
    
    # Place the same face image 6 times in the vertical strip
    face_names = ['right', 'left', 'back', 'front', 'down', 'up']
    for i in range(6):
        y_start = i * face_size
        y_end = y_start + face_size
        cubemap[y_start:y_end, :, :] = face_image

    return cubemap.flatten()


def blend_to_background(alpha, image, background):
    """Blend image with background using alpha blending."""
    if alpha == 1.0:
        return image
    elif alpha == 0.0:
        return background
    else:
        return (alpha * image.astype(np.float32)
                + (1. - alpha) * background.astype(np.float32)).astype(np.uint8)


class SafetyGymVideoBackground:
    """Video background distraction system for SafetyGym environments.
    
    This class handles dynamic video backgrounds that can be integrated
    with SafetyGym's existing color distraction system.
    """
    
    def __init__(self,
                 dataset_path=None,
                 dataset_videos=None,
                 video_alpha=1.0,
                 ground_plane_alpha=1.0,
                 num_videos=None,
                 dynamic=False,
                 seed=None,
                 shuffle_buffer_size=None):
        """Initialize video background system.
        
        Args:
            dataset_path: Path to DAVIS video dataset
            dataset_videos: List of video names or 'train'/'val'
            video_alpha: Alpha blending for video overlay [0,1]
            ground_plane_alpha: Alpha for ground plane transparency
            num_videos: Limit number of videos to use
            dynamic: Whether to animate through video frames
            seed: Random seed for reproducibility
            shuffle_buffer_size: Buffer size for shuffling frames
        """
        if not 0 <= video_alpha <= 1:
            raise ValueError('`video_alpha` must be in the range [0, 1]')
            
        self._video_alpha = video_alpha
        self._ground_plane_alpha = ground_plane_alpha
        self._random_state = np.random.RandomState(seed=seed)
        self._dynamic = dynamic
        self._shuffle_buffer_size = shuffle_buffer_size
        self._background = None
        self._current_img_index = 0
        self._step_direction = 1
        self._original_skybox_data = None
        self.sky_texture_id = None
        self._cached_mjr_context = None
        self._last_viewer = None
        self._texture_needs_upload = False
        self._previous_img_index = -1  # Track previous texture index for conditional GPU upload
        
        # Setup video paths
        if not dataset_path or num_videos == 0:
            self._video_paths = []
        else:
            # Use all videos if no specific ones were passed
            if not dataset_videos:
                if TF_AVAILABLE:
                    dataset_videos = sorted([fn.decode('utf-8') if isinstance(fn, bytes) else fn 
                                           for fn in tf.io.gfile.listdir(dataset_path)])
                else:
                    dataset_videos = sorted(os.listdir(dataset_path))
            # Replace video placeholders with actual video lists
            elif dataset_videos in ['train', 'training']:
                dataset_videos = DAVIS17_TRAINING_VIDEOS
            elif dataset_videos in ['val', 'validation']:
                dataset_videos = DAVIS17_VALIDATION_VIDEOS
                
            # Get complete paths for all videos
            video_paths = [
                os.path.join(dataset_path, subdir) for subdir in dataset_videos
            ]
            
            # Optionally limit number of videos
            if num_videos is not None:
                if num_videos > len(video_paths) or num_videos < 0:
                    raise ValueError(f'`num_videos` is {num_videos} but '
                                   'should not be larger than the number of available '
                                   f'video paths ({len(video_paths)}) and at least 0.')
                video_paths = video_paths[:num_videos]
                
            self._video_paths = video_paths
    
    def find_skybox_texture_id(self, model):
        """Find the texture ID of the skybox in the MuJoCo model."""
        for i in range(model.ntex):
            tex_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_TEXTURE, i)
            if tex_name == SKY_TEXTURE_NAME:
                return i
        return 0  # Fallback to first texture
    
    def initialize_background(self, model, data):
        """Initialize the video background system for a SafetyGym environment.
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
        """
        # Find skybox texture
        self.sky_texture_id = self.find_skybox_texture_id(model)
            
        # Store original skybox data for blending
        if self._original_skybox_data is None:
            sky_height = model.tex_height[self.sky_texture_id]
            sky_width = model.tex_width[self.sky_texture_id]
            sky_address = model.tex_adr[self.sky_texture_id]
                
            # MuJoCo stores texture data in tex_rgb (RGB format, 3 channels)
            sky_nchannel = 3
            sky_size = sky_height * sky_width * sky_nchannel
            # Get texture data from tex_rgb array
            self._original_skybox_data = model.tex_rgb[
                sky_address : sky_address + sky_size
            ].copy().astype(np.float32)
            
        self._reset_background(model, data)
            
    def reset_episode(self, model, data):
        """Reset background for new episode."""
        # Clean up old textures first
        self.cleanup_textures(model)
        # Then create new background
        self._reset_background(model, data)  
      
    def _reset_background(self, model, data):
        """Reset background for new episode."""
        if not self._video_paths:  # Skip if video backgrounds disabled
            return
            
        # Set the sky texture reference
        sky_height = model.tex_height[self.sky_texture_id]
        sky_width = model.tex_width[self.sky_texture_id]
        sky_address = model.tex_adr[self.sky_texture_id]
        
        # MuJoCo uses RGB format (3 channels)
        sky_nchannel = 3
        sky_size = sky_height * sky_width * sky_nchannel
        
        # Use original skybox as base
        sky_texture = self._original_skybox_data.copy()
        
        if self._video_paths:
            if self._shuffle_buffer_size:
                # Shuffle images from all videos together
                file_names = []
                for path in self._video_paths:
                    if TF_AVAILABLE:
                        video_files = tf.io.gfile.listdir(path)
                    else:
                        video_files = os.listdir(path)
                    file_names.extend([
                        os.path.join(path, fn.decode('utf-8') if isinstance(fn, bytes) else fn) 
                        for fn in video_files
                        if (fn.decode('utf-8') if isinstance(fn, bytes) else fn).lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])
                self._random_state.shuffle(file_names)
                file_names = file_names[:self._shuffle_buffer_size]
                images = [imread(fn) for fn in file_names]
            else:
                # Randomly pick a video and load all images
                video_path = self._random_state.choice(self._video_paths)
                if TF_AVAILABLE:
                    file_names = tf.io.gfile.listdir(video_path)
                else:
                    file_names = os.listdir(video_path)
                
                # Filter for image files
                file_names = [fn.decode('utf-8') if isinstance(fn, bytes) else fn for fn in file_names 
                            if (fn.decode('utf-8') if isinstance(fn, bytes) else fn).lower().endswith(('.png', '.jpg', '.jpeg'))]
                # Sort filenames numerically for consistent frame order
                file_names.sort(key=lambda x: int(x.split('.')[0]))
                if not self._dynamic:
                    # Pick a single static frame  
                    file_names = [self._random_state.choice(file_names)]
                
                images = [imread(os.path.join(video_path, fn)) for fn in file_names]
            
            # Pick random starting point and direction
            self._current_img_index = self._random_state.choice(len(images))
            self._step_direction = self._random_state.choice([-1, 1])
            
            # Prepare images in texture format
            texturized_images = []
            for image in images: # Generate all cubemap textures at start of episode
                is_cubemap_format = (sky_height == 6 * sky_width)
                if is_cubemap_format:
                    # Create cubemap with same image on all 6 faces
                    face_size = sky_width  # Each face is sky_width x sky_width (since height = 6 * width)
                    image_flattened = create_skybox_cubemap(image, face_size)    
                else:
                    # Not enough faces for a full cubemap, fall back to stretching
                    if len(image.shape) == 3 and image.shape[2] == 4:
                        image = image[:, :, :3]  # Remove alpha channel 
                    elif len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
                    image_flattened = size_and_flatten(image, sky_height, sky_width)
                new_texture = blend_to_background(self._video_alpha, image_flattened, sky_texture)
                # Store as uint8 to avoid type conversion during application
                # Ensure contiguous array for faster copying
                texture_array = np.ascontiguousarray(new_texture.astype(np.uint8))
                texturized_images.append(texture_array)
        else:
            self._current_img_index = 0
            # Ensure contiguous array for faster copying
            texturized_images = [np.ascontiguousarray(sky_texture.astype(np.uint8))]
        
        self._background = Texture(sky_size, sky_address, texturized_images)
        self._apply_background(model, None)  # No task available during reset
    


    def step_background(self, model, data, task=None):
        """Update background for dynamic mode."""
        try:
            if self._dynamic and self._video_paths and self._background:
                # Move forward/backward in the image sequence
                self._current_img_index += self._step_direction
                # Bounce at boundaries
                if self._current_img_index <= 0:
                    self._current_img_index = 0
                    self._step_direction = abs(self._step_direction)
                elif self._current_img_index >= len(self._background.textures):
                    self._current_img_index = len(self._background.textures) - 1
                    self._step_direction = -abs(self._step_direction)
                
                self._apply_background(model, task)
        except Exception as e:
            print(f"[VideoBackground] Error in step_background: {e}")
            # Disable video backgrounds on error
            self._video_paths = []
    

    def _apply_background(self, model, task=None):
        """Apply the background texture to the model for rendering."""
        if not self._background:
            return
            
        # Fast path: direct memory copy without type conversion
        start = self._background.address
        end = start + self._background.size
        texture = self._background.textures[self._current_img_index]
        
        # Optimized: Use numpy's copyto (benchmarked as fastest method)
        # Ensure both arrays are contiguous for optimal performance
        # model.tex_rgb is already contiguous (MuJoCo manages it)
        # texture is ensured to be contiguous during initialization
        np.copyto(model.tex_rgb[start:end], texture)
        
        # Only mark for GPU upload if texture index actually changed
        # This avoids unnecessary GPU operations when frame hasn't changed
        # (e.g., at boundaries where index bounces but stays the same)
        texture_changed = (self._current_img_index != self._previous_img_index)
        if texture_changed:
            self._texture_needs_upload = True
            self._previous_img_index = self._current_img_index
        
        # Try GPU upload only if texture changed and viewer is available
        # Note: We still copy to CPU memory every timestep (required for frame changes)
        # but only upload to GPU when the texture actually changed
        if texture_changed and task and hasattr(task, 'viewer') and task.viewer:
            self._try_gpu_upload_cached(model, task.viewer)
    
    def _try_gpu_upload_cached(self, model, viewer):
        """Optimized GPU upload with caching."""
        try:
            # Check if viewer changed or we need to refresh context
            if viewer != self._last_viewer or self._cached_mjr_context is None:
                self._last_viewer = viewer
                self._cached_mjr_context = None
                
                # Try to get MjrContext from viewer (cache for future use)
                if hasattr(viewer, 'ctx') and viewer.ctx is not None:
                    self._cached_mjr_context = viewer.ctx
                elif hasattr(viewer, 'con') and viewer.con is not None:
                    self._cached_mjr_context = viewer.con
                elif hasattr(viewer, '_mjr_context') and viewer._mjr_context is not None:
                    self._cached_mjr_context = viewer._mjr_context
            
            # Upload texture if we have a cached context and texture needs upload
            if self._cached_mjr_context is not None and self._texture_needs_upload:
                # Make context current if possible
                if hasattr(viewer, 'make_context_current'):
                    viewer.make_context_current()
                
                # Upload texture using cached context
                mujoco.mjr_uploadTexture(model, self._cached_mjr_context, int(self.sky_texture_id))
                self._texture_needs_upload = False  # Mark as uploaded
                
        except Exception as e:
            # Reset cache on error and continue
            self._cached_mjr_context = None
            self._last_viewer = None
            self._texture_needs_upload = True  # Keep trying next time 
    

    def cleanup_textures(self, model):
        """Clean up old textures and cubemaps from MuJoCo model."""
        try:
            
            # Clear background reference and free memory
            if self._background:
                # Explicitly delete texture arrays to free memory
                for i, texture in enumerate(self._background.textures):
                    del texture
                self._background = None
            
            # Reset original skybox data if we want to start fresh
            if self._original_skybox_data is not None:
                sky_height = model.tex_height[self.sky_texture_id]
                sky_width = model.tex_width[self.sky_texture_id]
                sky_address = model.tex_adr[self.sky_texture_id]
                sky_nchannel = 3
                sky_size = sky_height * sky_width * sky_nchannel
                # Restore original skybox data
                try:
                    model.tex_rgb[sky_address:sky_address + sky_size] = self._original_skybox_data.astype(np.uint8)
                except Exception as e:
                    print(f"[VideoBackground] Failed to restore original skybox: {e}")
            
            # Reset state variables
            self._current_img_index = 0
            self._step_direction = 1
            self._previous_img_index = -1  # Reset texture change tracking
            
            # Reset performance caches
            self._cached_mjr_context = None
            self._last_viewer = None
            self._texture_needs_upload = False
            
            # Force garbage collection to free memory
            gc.collect()
            
        except Exception as e:
            print(f"[VideoBackground] Error during texture cleanup: {e}")
    
    def shutdown(self, model):
        """Completely shutdown video background system and clean up all resources."""
        try:
            # Clean up textures
            self.cleanup_textures(model)
            
            # Clear all video paths to disable system
            self._video_paths = []
            
            # Clear original skybox data
            self._original_skybox_data = None
            
        except Exception as e:
            print(f"[VideoBackground] Error during shutdown: {e}")
    



def create_video_background_config(dataset_path, 
                                 video_alpha=0.7,
                                 dynamic=True,
                                 dataset_videos='train',
                                 num_videos=10,
                                 seed=None):
    """Create a video background configuration for SafetyGym.
    
    Args:
        dataset_path: Path to DAVIS dataset
        video_alpha: Blending alpha for video overlay
        dynamic: Whether to animate through frames
        dataset_videos: 'train', 'val', or list of video names
        num_videos: Number of videos to use
        seed: Random seed
        
    Returns:
        SafetyGymVideoBackground instance
    """
    return SafetyGymVideoBackground(
        dataset_path=dataset_path,
        dataset_videos=dataset_videos,
        video_alpha=video_alpha,
        dynamic=dynamic,
        num_videos=num_videos,
        seed=seed
    )
