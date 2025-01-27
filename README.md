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

## Abstract
We aim to build a robust system on a multi-llm architecture, capable of being specialized in any domain (given enough text data). The idea is to create a mixture-of-agents that debate and correct each other(Self Play Fine Tuning). Any “major learnings” where most of the agents are incorrect should be stored in a global memory (vector-db for now) that will influence the future responses of the agents. Eventually, the agents respond to a query if they are confident (Hallucinogens are not allowed :D ). Later, go on and fine-tune it to your use-case.

## Project Description