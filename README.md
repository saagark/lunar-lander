# Reinforcement learning for the toy game lunar-lander-v3

https://gymnasium.farama.org/environments/box2d/lunar_lander/

Using a pretty standard approach to solve this, using a Deep Q Network. I am using this project to learn about training RL agents. This is not meant to be optimal. 

## Results

Hyperparameters set in rl-lunar-lander.py 

### Performance over training run

![300 episodes][./gifs/episode3.gif]

![600 episodes][./gifs/episode6.gif]

![final agent][./gifs/final.gif]

Reward over most recent 100 episodes (tensorboard logging)
[./gifs/reward_avg100]


## Next steps/open questions

1. The reward/last 100 episodes initially goes down before increasing again. Need to investigate why this happens.
2. can add early exit once avg reward reaches a avg threshold >2xx points (200 points awarded for proper landing)
3. interested to know if theres a better defaul weight initialization
