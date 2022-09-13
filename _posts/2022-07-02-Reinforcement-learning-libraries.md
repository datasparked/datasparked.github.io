---
title:  "Part 9 : Reinforcement learning libraries"
excerpt: "Let's compare some reinforcement learning libraries"
header:
  teaser: /assets/images/header_images/inaki-del-olmo-NIJuEQw0RKg-unsplash.jpg
  overlay_image: /assets/images/header_images/inaki-del-olmo-NIJuEQw0RKg-unsplash.jpg
  overlay_filter: 0.5
  caption: "Photo credit: [**Iñaki del Olmo**](https://unsplash.com/@inakihxz?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText/)"
category:
  - reinforcement learning
---

In previous posts, we sometimes implemented manually some RL algorithms to explain how they work. However in practice, we will use some optimised libraries that implement common RL algorithms instead. In this post, we will compare some of these libraries. A list of such libraries can be found [here](https://github.com/godmoves/reinforcement_learning_collections).



## Main characteristics

| RL Libraries     | Framework                    | Tensorboard support     | Custom environment interface |
|------------------|------------------------------|-------------------------|------------------------------|
| [Keras-RL](https://github.com/keras-rl/keras-rl)         | Keras                        | No                      | No                           |
| [Tensorforce](https://github.com/tensorforce/tensorforce)      | Tensorflow                   | Yes                     | Yes                          |
| [OpenAI Baselines](https://github.com/openai/baselines) | Tensorflow                   | ?                       | No                           |
| [Stable baselines 3 ](https://github.com/DLR-RM/stable-baselines3) | Pytorch                   | Yes                     | Yes                          |
| [TF Agents](https://github.com/tensorflow/agents)       | Tensorflow                   | Yes                     | ?                            |
| [Ray / Rllib](https://github.com/ray-project/ray)      | Tensorflow / Pytorch / Keras | Yes                     | Yes                          |
| [Tensorlayer](https://github.com/tensorlayer/TensorLayer)      | Tensorflow                   | Yes                     | ?                            |
| [Rllab](https://github.com/rll/rllab) / [Garage](https://github.com/rlworkgroup/garage)   | Tensorflow / Pytorch         | ?                       | Yes                          |
| [Coach](https://github.com/IntelLabs/coach)            | TensorFlow                   | No but custom dashboard | Yes                          |


## Algorithms implemented

| RL Libraries     | DQN | DDPG | NAF / CDQN | CEM | SARSA | DqfD | PG / REINFORCE | PPO | A2C | A3C | TRPO | GAE | ACER | ACKTR | GAIL | SAC | TD3 | ERWR | NPO | REPS | TNPG | CMA-ES | MMC | PAL | TDM | RIG | Skew-Fit |
|------------------|-----|------|------------|-----|-------|------|----------------|-----|-----|-----|------|-----|------|-------|------|-----|-----|------|-----|------|------|--------|-----|-----|-----|-----|----------|
| Keras-RL         | X   | X    | X          | X   | X     |      |                |     |     |     |      |     |      |       |      |     |     |      |     |      |      |        |     |     |     |     |          |
| Tensorforce      | X   | X    | X          |     |       | X    | X              | X   | X   | X   | X    | X   |      |       |      |     |     |      |     |      |      |        |     |     |     |     |          |
| OpenAI Baselines | X   | X    |            |     |       |      |                | X   | X   |     | X    |     | X    | X     | X    |     |     |      |     |      |      |        |     |     |     |     |          |
| Stable baselines | X   | X    |            |     |       |      |                | X   | X   |     | X    |     | X    | X     | X    | X   | X   |      |     |      |      |        |     |     |     |     |          |
| TF Agents        | X   | X    |            |     |       |      | X              | X   |     |     |      |     |      |       |      | X   | X   |      |     |      |      |        |     |     |     |     |          |
| Ray / Rllib      | X   | X    |            |     |       |      | X              | X   | X   | X   |      |     |      |       |      | X   | X   |      |     |      |      |        |     |     |     |     |          |
| Tensorlayer      | X   | X    |            |     |       |      | X              | X   | X   | X   | X    |     |      |       |      | X   | X   |      |     |      |      |        |     |     |     |     |          |
| Rllab / Garage   | X   | X    |            | X   |       |      | X              | X   |     |     | X    |     |      |       |      |     | X   | X    | X   | X    | X    | X      |     |     |     |     |          |
| Coach            | X   | X    | X          |     |       |      | X              | X   |     | X   |      | X   | X    |       |      | X   | X   |      |     |      |      |        | X   | X   |     |     |          |
| Rlkit            | X   |      |            |     |       |      |                |     |     |     |      |     |      |       |      | X   | X   |      |     |      |      |        |     |     | X   | X   | X        |