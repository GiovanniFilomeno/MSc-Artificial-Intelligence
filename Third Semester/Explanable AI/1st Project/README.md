[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-VohRijK)

# Explainable AI Assignment 1: Projection Space Exploration
In this assignment, you are challenged to analyze and compare solutions of a problem, game, algorithm, model, or anything else that can be represented by sequential states. For this, you will project the high-dimensional states to the two-dimensional space, connect the states, and add meta-data to the visualization.

Exemplary solutions are provided in the `solution_rubik.ipynb` and `solution_2048.ipynb` notebooks. 

Further examples to analyze are (board) games and approximation algorithms. The 2048 notebook uses [OpenAI Gym](https://gym.openai.com/) to create a game environment and produce state data. There is a variety of first and third party environments for Gym that can be used.

## General Information Submission

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Group Members**

| Student ID    | First Name  | Last Name      | E-Mail |  Workload [%] |
| --------------|-------------|----------------|--------|---------------|
| k12119148        | Moritz      | Riedl         |k12119148@students.jku.at  |25%         |
| k12105068        | Verena      | Szojak         |verena@mail.at  |25%         |
| k12315325        | Giovanni      | Filomeno         |giovanni.filomeno.30@gmail.com  |25%         |
| k12105021        | Aaron      | Zettler         |zettler.aaron@gmail.com  |25%         |

### Dataset
Please add your dataset to the repository (or provide a link if it is too large) and answer the following questions about it:

* Which dataset are you using? What is it about?
* Where did you get this dataset from (i.e., source of the dataset)? How was the dataset generated?
* What is dataset size in terms of nodes, items, rows, columns, ...?
* What do you want to analyze?
* What are you expecting to see?

We created our own dataset and use the [Cliff Walking Environment](https://www.gymlibrary.dev/environments/toy_text/cliff_walking/), 
a toy enviornment provided in the gym library. It consists of 48 discrete states, 4 possible actions (up, down, left, right), 
and 2 discrete rewards (-1 at every step, -100 when falling of the cliff). We examine the trajectory data of 3 different 
reinforcement learning algorithms that use temporal difference learning, a combination of learning directly from raw 
experiency without knowing the environment dynamics and of updating estimates based on other estimates. In addition, we 
include a random policy that forces the agent to take a random action at every step. 
The algorithms work as follows: <br> <br>
**Sarsa:** an on-policy TD control method that learns values for state-action pairs that are stored in a q-table (q(s,a)). 
We choose an epsilon greedy action selection method and observe a reward and a next state based on the selected action. 
Then, we take another action epsilon greedily from the next state and update the q-table as follows: <br> 
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left\[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right\]$$ <br><br>
**Q-learning:** an off-policy TD control algorithm that learns state-action pairs by directly approximating the optimal
aciton-value function. The update rule for the q-table looks like this: <br>
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left\[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right\]$$ <br><br>
**Expected Sarsa:** similar to q-learning but instead of the maximum, the expected value is taken that considers the 
likelihood of each action under the given policy: <br>
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left\[ R_{t+1} + \gamma \sum_a \pi(a \mid S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t) \right\]$$ <br><br>
**Random:** a random policy from gym_gridworlds that samples an action randomly at every step and does not learn <br> <br>
For each policy we recorded 5000 episodes to ensure convergence and used the following hyperparameters:

| Algorithm        | Alpha  | Gamma | Epsilon |
|------------------|--------|-------|---------|
| Sarsa            | 0.1    | 0.999 | 0.5     |
| Q-learning       | 0.9    | 0.999 | 0.1     |
| Expected Sarsa   | 0.9    | 0.999 | 0.1     |
| Random           | 0.9    | 0.999 | 0.1     |

For our implementation of the algorithms and the assumptions about this environment, we followed the descriptions in
[Reinforcement Learning: An Introduction](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf) 
Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction (2nd ed.). The MIT Press.

Our dataset can be found [in this folder](data/cliff_walking). It consists of 338067 entries looking as follows:

| line        | cp  | algorithm | state | action | reward | next_state |
|------------------|--------|-------|---------|-----| ----| -----| 
| trajectory number [0, 4999] | start / intermediate / end   | Sarsa / Q-learning / Expected Sarsa / Random | between [0, 47] | [0, 1, 2, 3]| [-1, -100] | between [0, 47]

- line... index of the trajectory (from 0, 4999 for each of the 4 policies)
- cp... info about the current state (either the starting state, goal state or some intermediate state)
- algorithm... name of the algorithm
- state... current state the agent is in
- action... the selected action
- reward... the observed reward when taking the action
- next_state... the observed state after taking the action

Note, we will not use all 5000 trajectories for each algorithm for visualization to avoid that the plots get too crowded. 
How we will preprocess the data and what parts of the dataframe we will take is explained in detail in the final
submission notebook.

What do we want to analyze? <br>
We are interested in the behaviour of the different algorithms, especially the impact of the three different update rules
on the created trajectories. Consequently, we want to look at the learning behaviour of the agent in this environment
following different update rules. <br> 

We expect to see the following (inspired by Sutton and Barto 2018):
- Sarsa learns from the actions that the policy slects, also suboptimal actions. Thus, we get a more conservative policy
that results in a safer behaviour.
- Q-learning looks at the maximum Q-value over all actions for the next state. Thus, it is more greedy and always aims
for the best possible outcome.
- Expected SARSA is a trade-off between SARSA and Q-learning. It accounts for all possible actions and doesn’t focus on a single action.

Consequently, we hope to see the different behaviours reflected in the paths the algorithms take. Thus, when downprojecting
the data, we might find regions of trajectories that are mainly visited by one algorithm and avoided by others. 
Including the random policy in the visualizations could help to to detect differences when behaving randomly or according to 
a certain policy. Here we expect to see trajectory bundels for the learning algorithms but no bundels (apart form some coincidential ones) 
for the random bahviour. 

## Final Submission

* Make sure that you pushed your GitHub repository and not just committed it locally.
* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Create a single, clearly named notebook with your solution, e.g. solution.ipynb.
* Save your final executed notebook as html (File > Download as > HTML) and add them to your repository.

[The presentation can be found here](https://drive.google.com/file/d/13wN9rfia9eJH5LinJJz3smnOVY0pt_k1/view?usp=drive_link)
Alternatively, we uploaded the video and the presentation also in the repository. 

## Development Environment

Checkout this repo and change into the folder:
```
git clone https://github.com/jku-icg-classroom/xai_proj_space_2024-<GROUP_NAME>.git
cd xai_proj_space_2024-<GROUP_NAME>
```

Load the conda environment from the shared `environment.yml` file:
```
conda env create -f environment.yml
conda activate xai_proj_space
```

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:
```
jupyter lab
```

Go to http://localhost:8888/ and open the *template* notebook.

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.
