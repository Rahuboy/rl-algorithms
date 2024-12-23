The project contains the following files:

- `dqn.py`: Includes the main training loop for DQN
- `pg.py`: Includes the main training loop for Policy Gradient

- `environment_exploration.ipynb`: Includes the code for the random-agent exploration of the environments

- `mountain-car.ipynb`: The code that uses DQN to solve the mountain car environment
- `pong.ipynb`: The code that uses DQN to solve the pong environment
- `pong.py`: The file where the pong agent was trained
- `cartpole.ipynb`: The code that uses Policy Gradient to solve the cartpole environment
- `lunarlander.ipynb`: The code that uses Policy Gradient to solve the lunar lander environment

- `pong_curves`: svg plots for the losses and rewards of the pong agent
- `tensorboard_pong`: tensorboard logs for the pong agent

- `rewards_base_cartpole.pkl`: The rewards for the base cartpole agent
- `rewards_temporal_cartpole.pkl`: The rewards for the  cartpole agent that exploited temporal structure during policy gradients
- `rewards_advantage_cartpole.pkl`: The rewards for the  cartpole agent that exploited advantage function during policy gradients
- `rewards_base_lunarlander.pkl`: The rewards for the base lunar lander agent
- `rewards_temporal_lunarlander.pkl`: The rewards for the lunar lander agent that exploited temporal structure during policy gradients
- `rewards_advantage_lunarlander.pkl`: The rewards for the lunar lander agent that exploited advantage function during policy gradients
- `rewards_temporal_b2_lunarlander.pkl`: Same as temporal, but with a batch size of 2
