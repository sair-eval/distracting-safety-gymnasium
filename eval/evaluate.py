#!/usr/bin/env python3
"""Evaluate a trained SAIR actor in Distracting Safety Gymnasium at one or more distraction strengths.

Each level in --difficulties sets both the color-distraction strength beta_rgb (with --color_distractions) and
the opacity beta_video of the video background (with --video_distractions). One environment serves all levels.

Examples (see README.md for the checkpoints and the settings of each one):

    # SAIR trained with dynamic color distractions, evaluated across beta_rgb (Fig. 4)
    python evaluate.py --actor_path weights/sair/PointGoal1-color-seed1/actor_999999.pt \
        --domain_name SafetyPointGoal1-v0 --color_distractions dynamic --agent_filter agent_body \
        --difficulties 0 0.2 0.4 0.6 0.8 1.0 --num_episodes 300

    # the same policy with dynamic video backgrounds instead (DAVIS 2017, 30 training videos)
    python evaluate.py --actor_path ... --domain_name SafetyPointGoal1-v0 --video_distractions dynamic \
        --video_background_path /data/DAVIS/JPEGImages/480p --difficulties 1.0
"""
import argparse
import json
import os
import random

import numpy as np
import safety_gymnasium
import torch

from agent import utils
from agent.sac_ae import Actor, CostCritic

# Object types recoloured by the color distractions, as in training. 'agents' recolours the agent body and its
# heading marker; checkpoints trained before the 10 Oct 2026 environment update recoloured the body only, which
# --agent_filter agent_body reproduces (the README lists the setting per checkpoint).
RECOLOR_OBJECTS = ['hazards', 'walls', 'vases', 'agents', 'floor', 'goals', 'buttons', 'push_box', 'gremlins']
NUM_VIDEOS = 30          # the first 30 DAVIS-2017 training videos, as in training


def recolor_objects(agent_filter='agents'):
    return [agent_filter if o == 'agents' else o for o in RECOLOR_OBJECTS]


class SAIRAgent:
    def __init__(self, obs_shape, action_shape, device, hidden_dim=256, encoder_feature_dim=100, num_layers=4,
                 num_filters=32, log_std_min=-5, log_std_max=2, encoder_stride=1):
        self.device = device
        self.cfg = dict(hidden_dim=hidden_dim, encoder_feature_dim=encoder_feature_dim, num_layers=num_layers,
                        num_filters=num_filters, stride=encoder_stride)
        self.obs_shape, self.action_shape = obs_shape, action_shape
        self.actor = Actor(obs_shape, action_shape, hidden_dim, 'pixel', encoder_feature_dim, log_std_min,
                           log_std_max, num_layers, num_filters, encoder_stride).to(device).eval()
        self.cost_critic = None

    def load_actor(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Loaded actor from {path}')

    def load_cost_critic(self, path):
        c = self.cfg
        self.cost_critic = CostCritic(self.obs_shape, self.action_shape, c['hidden_dim'], 'pixel',
                                      c['encoder_feature_dim'], c['num_layers'], c['num_filters'],
                                      c['stride']).to(self.device).eval()
        self.cost_critic.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Loaded cost critic from {path}')

    def _obs(self, obs):
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        return obs.unsqueeze(0) if obs.dim() != 4 else obs

    @torch.no_grad()
    def select_action(self, obs):
        return self.actor(self._obs(obs)).cpu().numpy().flatten()

    @torch.no_grad()
    def cost_q(self, obs, action):
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.cost_critic(self._obs(obs), a))


def create_environment(domain_name, color_distractions, video_distractions, video_background_path, difficulty,
                       agent_filter, gremlin_num, frame_stack=3, action_repeat=4, image_size=64):
    video = video_distractions != 'none'
    config = {
        'change_geoms_color': color_distractions,
        'observe_vision': True,
        'vision_env_conf.vision_size': (image_size, image_size),
        'beta_rgb': difficulty,
        'object_filter': recolor_objects(agent_filter),
        'video_background_path': video_background_path if video else None,
        'video_dynamic': video_distractions == 'dynamic',
        'video_dataset': 'train',
        'num_videos': NUM_VIDEOS,
        'video_alpha': difficulty,
        **({'gremlins.num': gremlin_num} if gremlin_num is not None else {}),
    }
    env = safety_gymnasium.make(domain_name, render_mode='rgb_array', camera_name='vision_front_back', config=config)
    env = utils.FrameStack(env, k=frame_stack, resize_shape=(image_size, image_size), vision_key='vision_front_back')
    return utils.ActionRepeat(env, action_repeat)


def task_of(env):
    while hasattr(env, 'env'):
        env = env.env
    return env.unwrapped.task


def agent_xy(env):
    return [float(v) for v in task_of(env).agent.pos[:2]]


def video_frame(env, obs, cost):
    """The newest front view the policy sees, a fixed top-down view and a camera following the agent, upscaled 3x;
    a red bar marks steps with a cost."""
    obs = np.asarray(obs)
    size = obs.shape[0]
    task = task_of(env)
    views = [obs[:, :, 12:15].astype(np.uint8)] + [
        np.asarray(task.render(size, size, mode='rgb_array', camera_name=cam, cost={}), dtype=np.uint8)
        for cam in ('fixednear', 'track')]
    frame = np.repeat(np.repeat(np.concatenate(views, axis=1), 3, 0), 3, 1)
    if cost > 0:
        frame[:6] = (255, 0, 0)
    return frame


def save_gif(frames, path, fps=10):
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)


def evaluate_episode(env, agent, video=False, log_activity=False):
    obs = env.reset()
    frames = [video_frame(env, obs, 0)] if video else None
    xy, acts = ([agent_xy(env)], []) if log_activity else (None, None)
    done, ret, cost_sum, length, cost_steps = False, 0.0, 0.0, 0, 0
    step_cost_q, step_cost = [], []
    while not done and length < 1000:
        action = agent.select_action(obs)
        if agent.cost_critic is not None:
            step_cost_q.append(agent.cost_q(obs, action))
        obs, reward, cost, done, info = env.step(action)
        step_cost.append(float(cost))
        if log_activity:
            xy.append(agent_xy(env))
            acts.append([float(a) for a in np.asarray(action).ravel()])
        if frames is not None:
            frames.append(video_frame(env, obs, cost))
        ret += reward
        cost_sum += cost
        cost_steps += cost > 0
        length += 1
    result = dict(total_reward=float(ret), total_cost=float(cost_sum), episode_length=length,
                  cost_rate=cost_steps / length)
    if step_cost_q:
        result.update(step_cost_q=step_cost_q, step_cost=step_cost)
    if log_activity:
        p, a = np.asarray(xy), np.asarray(acts)
        result.update(path_length=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                      action_norm_mean=float(np.linalg.norm(a, axis=1).mean()), step_xy=xy, step_action=acts)
    return result, frames


def discounted_cost_to_go(costs, gamma):
    out, g = np.zeros(len(costs)), 0.0
    for t in reversed(range(len(costs))):
        g = costs[t] + gamma * g
        out[t] = g
    return out


def main():
    p = argparse.ArgumentParser(description='Evaluate a SAIR actor across distraction strengths')
    p.add_argument('--actor_path', required=True, help='actor_<step>.pt of a released checkpoint')
    p.add_argument('--cost_critic_path', default=None,
                   help='cost_critic_<step>.pt: also record the cost critic Q_C(o_t, a_t) and the cost at every step')
    p.add_argument('--cost_discount', default=0.99, type=float,
                   help='discount of the cost critic, for the summary line only (per checkpoint in weights/manifest.csv)')
    p.add_argument('--domain_name', default='SafetyPointGoal1-v0')
    p.add_argument('--color_distractions', default='none', choices=['none', 'static', 'dynamic'])
    p.add_argument('--video_distractions', default='none', choices=['none', 'static', 'dynamic'])
    p.add_argument('--video_background_path', default=None, help='DAVIS 2017 JPEGImages/480p folder')
    p.add_argument('--difficulties', nargs='+', type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                   help='distraction strengths: beta_rgb for color distractions, opacity for video backgrounds')
    p.add_argument('--agent_filter', default='agents', choices=['agents', 'agent_body'],
                   help="'agent_body' recolours the agent body but not its heading marker (checkpoints trained before 10 Oct 2026)")
    p.add_argument('--gremlin_num', default=None, type=int, help='number of gremlins in Button tasks (0 = none)')
    p.add_argument('--num_episodes', default=300, type=int)
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--video_episodes', default=0, type=int, help='save the first N episodes of every level as GIFs')
    p.add_argument('--log_activity', action='store_true',
                   help='record agent position and action per step, distance travelled and mean action norm per episode')
    p.add_argument('--output_dir', default='./eval_results')
    p.add_argument('--wandb_project', default=None, help='optional: log episodes to this W&B project (your own login)')
    args = p.parse_args()
    if args.video_distractions != 'none' and not args.video_background_path:
        p.error('--video_distractions needs --video_background_path')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = create_environment(args.domain_name, args.color_distractions, args.video_distractions,
                             args.video_background_path, args.difficulties[0], args.agent_filter, args.gremlin_num)
    agent = SAIRAgent(env.observation_space.shape, env.action_space.shape, device)
    agent.load_actor(args.actor_path)
    if args.cost_critic_path:
        agent.load_cost_critic(args.cost_critic_path)

    all_results = {}
    for difficulty in args.difficulties:
        print(f'\n=== difficulty {difficulty} ===')
        builder = env.unwrapped
        builder.beta_rgb = difficulty
        builder.video_alpha = difficulty
        if getattr(builder, 'video_background', None) is not None:
            builder.video_background._video_alpha = difficulty
        if getattr(builder, 'original_colors', None) is not None:
            builder.task.model.geom_rgba[:] = builder.original_colors   # undo the previous level's colours
        env.set_seed(args.seed)
        results = []
        for ep in range(args.num_episodes):
            result, frames = evaluate_episode(env, agent, video=ep < args.video_episodes, log_activity=args.log_activity)
            results.append(result)
            print(f"  episode {ep + 1}/{args.num_episodes}: return {result['total_reward']:.2f}, "
                  f"cost {result['total_cost']:.0f}")
            if frames is not None:
                path = os.path.join(args.output_dir, f'difficulty{difficulty:.1f}_ep{ep}.gif')
                save_gif(frames, path)
                if sum(int(f[6:].max() == 0) for f in frames):
                    print('  WARNING: some video frames are all black')
            if run is not None:
                run.log({'return': result['total_reward'], 'cost': result['total_cost'], 'difficulty': difficulty})
        rets = np.array([r['total_reward'] for r in results])
        costs = np.array([r['total_cost'] for r in results])
        stats = dict(num_episodes=len(results), avg_reward=float(rets.mean()), std_reward=float(rets.std()),
                     avg_cost=float(costs.mean()), std_cost=float(costs.std()),
                     frac_cost_over_25=float((costs > 25).mean()))
        if args.log_activity:
            stats['avg_path_length'] = float(np.mean([r['path_length'] for r in results]))
        if agent.cost_critic is not None:
            q = np.concatenate([r['step_cost_q'] for r in results])
            g = np.concatenate([discounted_cost_to_go(r['step_cost'], args.cost_discount) for r in results])
            stats['cost_q_mean'], stats['cost_to_go_mean'] = float(q.mean()), float(g.mean())
        all_results[str(difficulty)] = dict(results=results, statistics=stats)
        print('  ' + ', '.join(f'{k} {v:.3f}' for k, v in stats.items() if isinstance(v, float)))

    path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(path, 'w') as f:
        json.dump(all_results, f)
    print(f'\nresults: {path}')
    if run is not None:
        run.finish()


if __name__ == '__main__':
    main()
