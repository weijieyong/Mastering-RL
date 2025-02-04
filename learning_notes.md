# Reinforcement Learning (RL) learning notes

<p align="center">
  <img src="https://gymnasium.farama.org/_images/AE_loop.png" width="70%">
  <br>
  <em>agent-environment-loop</em>
</p>

> **Goal of RL:** To find the *optimal policy* that maximizes the expected cumulative reward.
> 

Core RL problem: **Agent-Environment Interaction**

- Agent: learner, decision maker, (robot)
- Environment: everything else that agent interacts with (workspace, objects, etc )
- State (Observation in gymnasium): agent’s perception of the env at given moment
    - *! should contain enough information for the agent to make a good decision.*
- Action: what agent can do to the env. (e.g. robot moving gripper/joints)
- Reward: *scalar* signal from env → agent after each action
    - positive reward: encourage desired behaviour
    - negative reward(penalty): discourage undesired behaviour
    - sparse rewards: given only for achieving the final goal
    - dense rewards: given more frequently, during intermediate steps, for faster learning
- Timestep: action → env → state → reward loop, repeats in a sequence (episoide / trajectory)

**Markov Decision Processes (MDPs):**

- goal: find a policy that maximizes the cumulative reward over time
- future state depends only on the *current* state and action (ignore history)

**Value Functions**

- help agent estimate how “good”(future rewards) is it to be in a state, or take an action in a state.
- important to guide agent towards states and action that lead to higher future rewards
- state-value function: "How good is it to be in state S?”
- action-value function: "How good is it to take action A in state S?”

**Policies**

- function that maps states to actions, agent’s brain / strategy
- deterministic policy: For each state, it always chooses the same action.
- stochastic policy: For each state, it gives a probability distribution over actions.

**Exploration vs Exploitation**

- exploration: trying out new actions to discover better states and rewards. "exploring the environment."
- exploitation: using the current policy to choose actions that are believed to be good based on past experience. "exploiting what you already know."
- we need to balance exploration and exploitation. too much exploration might slow down progress, too much exploitation might get you stuck in a suboptimal solution.


## Useful resources for starting
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)

### Repos
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Single file implementation of Deep Reinforcement Learning algorithms
- [Tianshou](https://github.com/thu-ml/tianshou) - Tianshou (天授) is a reinforcement learning platform based on pure PyTorch.

