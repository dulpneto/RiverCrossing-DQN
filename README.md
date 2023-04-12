# RiverCrossing-DQN

## Implementation of River Crossing Domain 

## Exponential Utility Theory
- TD - Applying exponential utility theory on QLearning tempora difference
- Target - Applying exponential utility theory on QLearning target
- LSE - Applying LogSumExp strategy on target update to minimize overflow errors

This can be changed on argument bellman_update (-b)

## Risk Attitude
It can be controlled change lambda argument (-l)
- Negative values are risk averse
- Zero is risk neutral
- Positive risk prone

## Implementation of QLearning
Run: python3 DQNRunner.py -t QL -b Target -l -1.0

## Implementation of DeepQNetwork
Run: python3 DQNRunner.py -t DQN -b Target -l -1.0

## Alternative DeepQNetwork Implementation
### DQN with table cache
It refreshes a Q-table every time the target model is trained. It is useful to speed the tests

Run: python3 DQNRunner.py -t DQN_CACHED -b Target -l -1.0


### DQN without model usage
It creates an Q-Table and use it to learning, model is trained by not used to update the Q-table. This implementation helps to find model approximation errors.

Run: python3 DQNRunner.py -t DQN_SKIP -b Target -l -1.0

When running this implementation two Value tables are plotted. One from Q-table and one from model