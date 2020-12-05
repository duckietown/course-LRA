# Learning-based Control - Data Collection {#lra2-model-based-learning status=draft}

Excerpt: Collect Duckiebot data to train a model of the Duckiebot.

<div class='requirements' markdown='1'>

  Requires: [Docker Basics](+duckietown-robotics-development#docker-basics)

  Requires: Knowledge about machine learning

  Requires: Some knowledge of control theory

  Results: Implement an LQR Controller using learning for your Duckiebot.

</div>

<minitoc/>

## Overview of the task

For this task, we will be using the Duckietown Gym to run standard lane following in simulation and obtain data of the Duckiebot model. We will collect the information about the state $\vec{x}=\begin{bmatrix} d& \varphi \end{bmatrix}^\intercal$, and the control input $u$, where $x \in \mathcal{R}^2$ and $u \in \mathcal{R}$.

  - $d$ is the distance of the center of the Duckiebot's axle to the center of the right lane.
  - $\varphi$ is the angle (rads) from centerline to heading of the Duckiebot.
  - $u$ is the steering input (in radians per second).


## Duckietown Gym

We will be working in the Duckietown Gym. Gym-Duckietown is a simulator for the Duckietown Universe, written in pure Python/OpenGL (Pyglet). It places your agent, a Duckiebot, inside of an instance of a Duckietown: a loop of roads with turns, intersections, obstacles, Duckie pedestrians, and other Duckiebots. If not yet familiar, please read the docs directly on the [Gym-Duckietown](https://github.com/duckietown/gym-duckietown) to get a better understanding about working with the simulator.

For this exercise, however, we will be using a containerized docker image with everything you need to collect the data, train the machine learning model, and develop your own controller.

To run the Docker image, run the following command in your local machine.

    laptop $ Docker run..... #TODO Vincenzo add command.

Once installed, navigate to the `gym-duckietown/exercises/LRA` directory. This is where you will develop your solution. You will find Jupyter-notebooks for model training and sample collection. In addition, the `lraClass.py` file has been included to facilitate loading and saving your data.

## Data collection

### Collecting the data using default Controller

The first task is to collect the state and control input data. For this, you will need to create an Gym instance of the Duckietown Environment. An example of this setup procedure is shown below. The function `env.render()` displays a graphical representation of the environment to your screen.  

```python
from gym_duckietown.envs import DuckietownEnv
#   Create the environment

env = DuckietownEnv(
    map_name = "udem1",
    domain_rand = False,
    draw_bbox = False
)

obs = env.reset()
env.render()
```

You can use the default controller `basic_control.py` to make your Duckiebot move. In the while loop, you will need to do two things:

1. Extract the steering input (in radians per second)
2. Collect the data.

Run your simulation until it finished or crashes, and then save the data to a csv file.

Tip: For data collection, you can use the LRA helper function `collectData(u,x)` which takes inputs a scalar $u$ and a vector $\vec{x}$.

Tip: For saving your data, you can use the LRA helper function `saveData("filepath.csv")`.

You should now have a `.csv` file containing your training data. An example of what this file should look like is shown below:

```python
d,phi,u
-0.1618443443591879,-0.298587790613436,-1.6330326457217237
-0.16389157947457456,-0.29663545184459117,-1.632756126910639
-0.16590270172174604,-0.28918014534655817,-1.6122521921396433
-0.17345815224603006,-0.25618446121623173,-1.5169998357130914
-0.17518546625467168,-0.2396585301709049,-1.4624035526137578
-0.1767672947637351,-0.21713463020459525,-1.3854049199878498
...
...
```

### Collecting the data with random control signals

Repeat the process you just did, but this time use random control signals that are uncorrelated to the states. Run the simulation and record the states, and control input.
Save the file for your random control input as a separate `.csv` file.


## Model Training

The next step is to use the data you collected to develop a linear model of the Duckiebot. The model can be written as follows:

$$ \vec{X}_{t+1} = A\vec{X}_t+B\vec{u}_t $$

Where,
* $ \vec{X}_{t+1} \in \mathcal{R}^{2x2} $
* $ \vec{u}_{t} \in \mathcal{R}^{2x1} $

The goal is to find the matrices $A$ and $B$.

### Data Cleaning

Not all the collected data might be relevant, it is your task now to determine what data to include and why.

The data can be easily loaded using the LRA helper function `loadData("data.csv")`.

```python
lra = LRA2_HELPER()
data = lra.loadData("model_data.csv")
```

Hint: the data is imported as a pandas DataFrame. This can easily be sliced, truncated, sorted, and allows for other operations. You should look at the `.iloc[]` and `.loc[]` methods to facilitate data cleaning and pre-processing.

NOTE: Note that an important factor is that the control signal should be bound by a magnitude of 1. $$|u| < 1 $$

### Model training

You are free to use any model you want for training.  The goal is to use regression to find $A$ and $B$. One way to do this is to use `sklearn.linear_model` library to initialize a model.

In `sklearn` models follow the convention of X,Y, where X represents the training data, and Y the outputs. The model you are trying to predict is shown below: Keep in mind that we are fitting a linear model.

$\begin{bmatrix} d_{t+1} \ \varphi_{t+1} \end{bmatrix} = \begin{bmatrix} A& | B \end{bmatrix} \begin{bmatrix} d_t \ \varphi_t \ u \end{bmatrix} $

Deliverable: Submit your A and B matrices for the data collected using the PID controller.

Deliverable: Submit your A and B matrices for the data collected using random control signals.

Deliverable: What are the advantages and disadvantages of using the data collected using the PID controller?

Deliverable: What are the advantages and disadvantages of using the data collected using the random control signals?

Deliverable: Is all the data collected useful? Explain why or why not. In addition, elaborate on any type of data pre-processing you might have performed.

## Linear Quadratic Regulator - Control

Now that you have learned the model of the Duckiebot using machine learning, the objective is to implement a controller to test how accurate the model prediction is. To do this you will use a Linear Quadratic Regulator (LQR).

The LQR formulation is shown below:

\[
J = \int_{0}^{\inf}(\vec{x}^T Q \vec{x} + \vec{u}^T R \vec{u}) \,dt
\]

The LQR is an optimization problem that penalizes the state and control inputs according to the matrices $Q$ and $R$, respectively. You will need to solve the discrete [Algebraic Riccati Equation](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.solve_discrete_are.html)

The solution of the ARE can be used to obtain the gains. Compute the gain matrix K and implement it in your control. Run the simulator again, and record your screen for your Duckiebot running on your custom model-based-learning controller.


Deliverable: A short video (~5-10 sec) of your simulated Duckiebot running on your LQR controller.

Deliverable: Your Q and R matrices

Deliverable: Your K matrix

Deliverable: Question: How does the performance of your controller compare to the controller you used to collect data initially?

Deliverable: Question: Are you penalizing control input, why?

Deliverable: Question: Are you penalizing the states, why?
