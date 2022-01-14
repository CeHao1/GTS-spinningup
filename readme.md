**Status:** Active (under active development, breaking changes may occur)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!

# commands
The command for the run is
python3 spinup/algos/sac/sac.py "192.168.1.14" "192.168.1.11" "192.168.1.10" "192.168.1.12" "192.168.1.23" --gamma 0.98 --start_steps 160000 --seed 2 --replay_size 4000000 --exp_name "TD 5 learning while sampling with evaluation in parallel gamma 098" --evaluate True

The IP addresses should be rewritten accordingly.
In this case, 4 PS4s ("192.168.1.14" "192.168.1.11" "192.168.1.10" "192.168.1.12") are used for training and the last one "192.168.1.23" is used for evaluation. In your case, with only one PlayStation, you need to set "--evaluate False" and concentrate on the training only. You can evaluate the trained models later.

And if I remember correctly, scikit-learn (0.23.1) didn't work with the code for some backward incompatibilities. So I needed to use the older version by fixing the version of scikit-learn to 0.22.2 by:
pip3 install scikit-learn==0.22.2


python3 spinup/algos/sac/sac.py "192.168.124.5" --gamma 0.98 --start_steps 160000 --seed 2 --replay_size 4000000 --exp_name "TD 5 learning while sampling with evaluation in parallel gamma 098" 
