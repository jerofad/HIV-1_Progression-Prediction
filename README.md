# HIV-1_Progression-Prediction
A kaggle competition to predict the likelihood that an HIV patient's infection will become less severe, given a small dataset and limited clinical information.

# Getting started
To be able to repeat the same experiments, you are going to need a laptop with Miniconda (a minimal version of Anaconda) and several Python packages installed which are listed in the requirements.txt file.
Following instruction would work as is for Mac or Ubuntu linux users, Windows users would need to install and work in the Gitbash terminal.

## Download and install Miniconda
Please go to the [Anaconda website](https://conda.io/miniconda.html).
Download and install *the latest* Miniconda version for *Python* 3.6 for your operating system.

```bash
wget <http:// link to miniconda>
sh <miniconda .sh>
```

After that, type:

```bash
conda --help
```

and read the manual.

## Create a clone of this repository 
You can either create a clone or make a download of this repository. To make a clone, simply use the command below on your terminal:

```bash
git clone https://github.com/jerofad/HIV-1_Progression-Prediction.git
```

If you are not using git, just download the repository as zip, and unpack it:

```bash
wget https://github.com/jerofad/HIV-1_Progression-Prediction/archive/master.zip
For Mac users:
curl -O https://github.com/jerofad/HIV-1_Progression-Prediction/archive/master.zip
unzip master.zip
```
Change into the course folder, then type:

```bash
cd HIV-1_Progression-Prediction
```
## # To perform Data Analysis
Run the command below:

```bash
python3 data_analysis.py
```
The results is saved into the directory named Figures.

## # To Repeat the experiment on all 10 Models
Run the command below:

```bash
python3 models.py
```
It saves the result into a text file named "all_models_report.txt"
## # Tuning the Hyper-Parameters
If you want to train the top 5 models with parameters tuning, Run the command below 

```bash
python3 parameter_tuning.py
```
the result of the models can be seen in the text file "models_report.txt"
## # Final top 5 Models
Run the command below:

```bash
python3 train.py
```
the result of the models can be seen in the text file "models_report.txt"

