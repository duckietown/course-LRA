# [LRA1] Imitation Learning {#part:lra-1 status=ready}

Excerpt: Train your network by imitation.

In this part you will try to train a network via imitation, and consequently test its performance.

<div class='requirements' markdown='1'>

  Requires: Some theory about machine learning

  Requires: A proper laptop setup.

  Results: Get a feeling of what imitation learning is.

</div>


<minitoc/>



## Learning imitation learning {status=ready}

#### Imitation Learning {#exercise:imitation_learning}

For this exercise, you will get a repository that takes care of creating the environment for you.
You will be required to modify a Jupyter Notebook file only, as the rest is fairly complex.

Clone the template repository:

    laptop $ git clone --recursive git@github.com:duckietown-ethz/lra-exercise.git

The `--recursive` option is important to also clone submodules, if you forgot it, run

    laptop $ git submodule init

    laptop $ git submodule update

Inside the repository, requirements

    laptop $ docker-compose build
    laptop $ docker-compose up

Note: The build command, will require a lot of time, but you only need to build it once.

A lot of text will pop up, connect to the notebook server, to do this, you should see an address of the form `http://127.0.0.1:8888/...`, just open the link in a web browser. Once connected enter in the notebook directory and open the file `01-duckietown-imitation-learning-ipynb`, then follow the instructions in it.

When you want to interrupt the containers, press <kbd>Ctrl</kbd>-<kbd>C</kbd> then wait for it to gracefully stop. This will prevent many problems.
