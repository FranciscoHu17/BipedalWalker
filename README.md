# CSE352_Final_Project

## Dependencies
Python version 3.5 to 3.8

### If on Windows: 
- Install Anaconda at https://www.anaconda.com/products/individual
- Verify your installation in Anaconda Prompt with the command ```conda list``` which should display a list of installed packages
- Also verify that python is working by entering the command ```python```
- Run ```conda install swig```
- Make sure you have Microsoft Visual C++ 14.0 or greater. It can be installed at https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install ```gym``` by running
```
git clone https://github.com/openai/gym
cd gym
pip install -e .[box2d]
```
- Install ```pytorch``` by runnning ```conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch```


### If on MacOS (has not been tested, but should theoretically work):
- Install Anaconda at https://www.anaconda.com/products/individual
- Verify your installation in Anaconda Prompt with the command ```conda list``` which should display a list of installed packages
- Also verify that python is working by entering the command ```python```
- Run ```conda install swig```
- Install ```gym``` by running
```
git clone https://github.com/openai/gym
cd gym
pip install -e .[box2d]
```
- Install ```pytorch``` by runnning ```conda install pytorch torchvision torchaudio -c pytorch```


### If on Linux:
- Install ```gym``` by running
```
git clone https://github.com/openai/gym
cd gym
pip install -e .[box2d]
```
- Install ```pytorch``` by runnning ```pip3 install torch torchvision torchaudio```


More information on installing OpenAI Gym API at https://github.com/openai/gym#installation


## Running the Code:
- Q-Learning
```
# From the root directory:
cd Q-Learning
python QLearningWalker.py
```
- DQN
```
# From the root directory:
cd DQN
python dqn_walker.py
```
- TD3:
```
# From the root directory:
cd TD3
python td3_walker.py
```

## Report
We evaluated the performances of our algorithms with a comprehensive academic report. This could be found at https://github.com/FranciscoHu17/BipedalWalker/blob/main/BipedalWalkerReport.pdf
