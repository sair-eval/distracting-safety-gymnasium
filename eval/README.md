# Evaluating the SAIR checkpoints

This folder holds the evaluation code and the trained policies of the paper "Safety-Aware Invariant Representations
for Reinforcement Learning from Pixels". The code builds on [Deep Bisimulation for Control](https://github.com/facebookresearch/deep_bisim4control).

## Checkpoints

`weights/` holds the final checkpoints (1M environment steps) of the SAIR and SAIR ($w_c = 0$) runs in the
training curves of the paper: four tasks (SafetyPointGoal1, SafetyCarGoal1, SafetyPointButton1, SafetyPointGoal2),
trained without distractions (`none`) or with dynamic color distractions (`color`, $\beta_{rgb} = 1$), seeds 1 to 3.
It also holds the SafetyPointGoal1 policies of SAIR, SAIR ($w_c = 0$) and DBC trained with dynamic video backgrounds
(`video`, $\beta_{video} = 1$, one seed). Each folder holds the actor (`actor_999999.pt`), the cost critic
(`cost_critic_999999.pt`, not for DBC, which has none) and their `MD5SUMS`. For DBC there is also one policy per
condition on SafetyPointGoal1, SafetyCarGoal1 and SafetyPointGoal2 (`actor.pt`), whose training seed was not recorded.

```
weights/
  manifest.csv                     evaluation flags per checkpoint (--agent_filter, --gremlin_num, --cost_discount)
  sair/<Task>-<none|color>-seed<1-3>/    sair/PointGoal1-video-seed1/
  sair_wc0/<Task>-<none|color>-seed<1-3>/    sair_wc0/PointGoal1-video-seed1/
  dbc/PointGoal1-video-seed1/    dbc/<Task>-<none|color>/
```



## Installation

With Docker, from the repository root:

```bash
docker build -f eval/Dockerfile -t sair-eval .
docker run --gpus all -it sair-eval        # starts in /app/eval, DAVIS 2017 in /data/DAVIS
```

Without Docker (Python 3.8), from the repository root:

```bash
pip install -e .
pip install -r eval/requirements.txt
# video backgrounds only: DAVIS 2017
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip && unzip DAVIS-2017-trainval-480p.zip
```

Rendering uses MuJoCo with `MUJOCO_GL=egl` on a GPU; on a machine without one, use `MUJOCO_GL=osmesa`.

## Usage

Run from this folder. Each value in `--difficulties` sets the strength of the active distraction: $\beta_{rgb}$ for
color distractions, the opacity $\beta_{video}$ for video backgrounds.

Final return and cost of a policy at its training condition (300 episodes):

```bash
python evaluate.py --actor_path weights/sair/PointGoal1-color-seed1/actor_999999.pt \
    --domain_name SafetyPointGoal1-v0 --color_distractions dynamic --agent_filter agent_body \
    --difficulties 1.0 --num_episodes 300
```

Sweep over the color distraction strength, as in the distraction-strength figure:

```bash
python evaluate.py --actor_path weights/sair/PointGoal1-color-seed1/actor_999999.pt \
    --domain_name SafetyPointGoal1-v0 --color_distractions dynamic --agent_filter agent_body \
    --difficulties 0 0.2 0.4 0.6 0.8 1.0 --num_episodes 300
```

Dynamic video backgrounds (the first 30 DAVIS-2017 training videos, as in training):

```bash
python evaluate.py --actor_path weights/sair/PointGoal1-none-seed1/actor_999999.pt \
    --domain_name SafetyPointGoal1-v0 --video_distractions dynamic \
    --video_background_path /data/DAVIS/JPEGImages/480p --difficulties 0 0.2 0.4 0.6 0.8 1.0
```

A policy trained with video backgrounds, at its training condition:

```bash
python evaluate.py --actor_path weights/sair/PointGoal1-video-seed1/actor_999999.pt \
    --domain_name SafetyPointGoal1-v0 --video_distractions dynamic \
    --video_background_path /data/DAVIS/JPEGImages/480p --difficulties 1.0 --num_episodes 300
```

Optional:
- `--cost_critic_path weights/.../cost_critic_999999.pt` records the cost critic's estimate $Q_C(o_t, a_t)$ and the
  cost at every step (with `--cost_discount` from the manifest).
- `--log_activity` records the agent's position and action at every step, and the distance travelled per episode.
- `--video_episodes N` saves the first N episodes of every level as GIFs (policy input, top-down view, follow camera).

Results are written to `--output_dir` (default `./eval_results/evaluation_results.json`): per level, every
episode's return, cost, length and the optional per-step records, plus summary statistics (mean return and cost,
fraction of episodes with cost above the budget of 25).

The policy acts with its mean action. Episodes last 1000 environment steps (250 agent steps with action repeat 4),
and the observation stacks three front and back camera frames of 64 x 64 pixels.
