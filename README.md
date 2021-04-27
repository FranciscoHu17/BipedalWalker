# CSE352_Final_Project

## How to run:
If you're on MacOS, following the installation steps on Windows should work, but it has not been tested yet.

If on Windows: 
- Install Anaconda at https://www.anaconda.com/products/individual
- Verify your installation in Anaconda Prompt with the command ```conda list``` which should display a list of installed packages
- Also verify that python is working by entering the command ```python```
- Run ```conda install swig```
- Make sure you have Microsoft Visual C++ 14.0 or greater
- Install ```gym``` by running
```
git clone https://github.com/openai/gym
cd gym
pip install -e .[box2d]
```
- Install ```PyTorch``` by runnning ```py -m pip install torch```
- Navigate to the location of the walker.py file and run ```py walker.py```

If on Linux:
- Install ```gym``` by running
```
git clone https://github.com/openai/gym
cd gym
pip install -e .[box2d]
```
- Install ```PyTorch``` by running ```pip install torch```
- Navigate to the location of the walker.py file and run ```python walker.py```

More information on installing OpenAI Gym API at https://github.com/openai/gym#installation
