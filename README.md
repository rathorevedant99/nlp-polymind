# PolyMind NLP Project
### Year: 2024-25

## Setup
1. Clone the repository
```bash

git clone git@github.com:rathorevedant99/nlp-polymind.git
```
2. Move to the project directory
```bash
cd nlp-polymind
```
3. Create a virtual environment
```bash
python3 -m venv .env
```
4. Activate the virtual environment
```bash
source .env/bin/activate
```
5. Install the dependencies
```bash
pip install -r requirements.txt
```
6. Checkout to your development branch
```bash
git checkout -b <branch-name>
```
7. Have fun developing!

## How this works
- Repo structure is as follows:
    - `src`: Source code
        - `agent`: Agent classes
            - `base.py`: Base class for all agents
            - `critic.py`: Critic class
            - `expert.py`: Expert class
            - `team.py`: Team class (contains the **expert** team)
        - `data.py`: Data classes
        - `eval.py`: Evaluation classes (contains the **debate** class)
    - `configs`: Configuration files
    - `data`: Data files
    - `results`: Results files (LORA weights, etc.)
    - `outputs`: Output files (Debate logs, etc.)
    - `main.py`: Main file to run the project
    - `requirements.txt`: Dependencies
    - `README.md`: This file

- For config.yaml to be used, ensure Hydra is installed
```bash
pip install hydra-core
```

- To run the project, use the following command:
```bash
python main.py
```

- To run the project with a specific config, use the following command:
```bash
python main.py arg_name=arg_value
```
Example:
```bash
python main.py lora.r=8
```
This will use a r=8 LORA for the experts and all other parameters will be default.

All work to be done on the repo can be found in the wiki page:
[Work to be done!](https://github.com/rathorevedant99/nlp-polymind/wiki/Work-to-be-done!)

## Abstract
We aim to build a robust system on a multi-llm architecture, capable of being specialized in any domain (given enough text data). The idea is to create a mixture-of-agents that debate and correct each other(Self Play Fine Tuning). Any “major learnings” where most of the agents are incorrect should be stored in a global memory (vector-db for now) that will influence the future responses of the agents. Eventually, the agents respond to a query if they are confident (Hallucinogens are not allowed :D ). Later, go on and fine-tune it to your use-case.

## Project Description
Important Links:
[SPIN Paper](https://arxiv.org/abs/2401.01335)
[GitHub Repo](https://github.com/uclaml/SPIN)  
