# Growing Artificial Neural Networks for Control: the Role of Neuronal Diversity

This is the code for our paper submission to Gecco '24.


<div style="text-align:center;">

![gecco2024.gif](..%2F..%2F..%2F..%2FLibrary%2FMobile%20Documents%2Fcom%7Eapple%7EKeynote%2FDocuments%2Fgecco2024.gif)

*Visualization the trajectories learned by our model (corresponding to the results presented in Figure 3 and Table 1 of our paper)*
</div>

## How to install

To install all required python packages you can create a new conda environment from the provided requirements file:

`conda env create -f environment.yml`

We ran our code on a Linux server with an NVIDIA RTX A6000 GPU.


## How to run simulations

Under folder examples you can find scripts for reproducing the experiments and figures presented in the paper.

When running our model, the Neural Developmental Program (NDP), you have the option of activating intrinsic hidden states and/0r lateral inhibition.

Script examples/NDP/train.py contains code for training an agent in different locomotion tasks (ant, halfcheetah, double inverted pendulum and reacher) with three configurations of the NDP: 

- NDP: intrinsic hidden states are activated and inhibition is deactivated
- NDP-vanilla: both intrinsic hidden states and inhibition are deactivated
- NDP-inhib: intrinsic hidden states are deactivate and inhibition is activated


To train the Hypernet baseline you can run script examples/hypernet/train.py

## How to visualize results

During training, data files will be saved under directory projects. A unique directory will be created for each experiment.

You can create visualizations for each project by calling the examples/NDP/post_process.py script, specifying the name of the project.

You can also monitor results during training using wandb. To do this, you need to change the log flag in the default_config.yaml file to 1.



