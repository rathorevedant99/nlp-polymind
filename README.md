# PolyMind NLP Project

## Abstract
Adapting Super Tiny Language Models
(STLMs) to specialized tasks remains chal-
lenging due to their limited capacity and
compute budgets. We propose a memory-
augmented critique framework that en-
ables collaborative, self-supervised refine-
ment of STLMs through structured feed-
back. Our approach orchestrates mul-
tiple LoRA-tuned expert models along-
side a separate judge model, leveraging
a persistent memory module to store cri-
tiques. These critiques are subsequently
transformed into instruction-tuning prompts
for second-stage fine-tuning. Evaluated
on dialogue summarisation (SAMSum), our
method achieves statistically significant per-
formance improvements: average ROUGE-
1 F1 increases from 0.393 to 0.406 in the
3-expert setup (p < 0.05), with 66.7% of
experts showing individual gains. Addition-
ally, we identify a maximum memory size
threshold 90% of the evaluation set, beyond
which overfitting occurs. The results sug-
gest that our critique-driven architecture of-
fers a scalable, supervision-free alternative
to standard fine-tuning for STLMs.
## Overview

PolyMind implements a mixture-of-agents architecture where multiple expert models are fine-tuned on specific domains and correct each other through shared memory. The memory is created with the expert's responses and the critic's feedback. The architecture includes:

- Multiple expert agents that can be fine-tuned on specific domains
- A critic agent that evaluates expert responses
- A debate mechanism for self-improvement
- Memory-augmented learning using simple dictionary
- Statistical analysis of agent performance

## Architecture

The project structure is organized as follows:

```
nlp-polymind/
├── src/
│   ├── agent/           # Agent implementations
│   │   ├── base.py     # Base agent class
│   │   ├── expert.py   # Expert agent implementation
│   │   ├── critic.py   # Critic agent implementation
│   │   └── team.py     # Expert team management
│   ├── data.py         # Data handling
│   ├── eval.py         # Evaluation and debate
│   ├── memory.py       # Memory management
│   ├── metrics.py      # Performance metrics
│   └── utils/          # Utility functions
├── configs/            # Configuration files
├── data/              # Data cache storage
├── results/           # Model weights and results
├── outputs/           # Debate logs and outputs
└── experiments/       # Experimental scripts
```

## Features

- **Multi-Agent System**: Implements expert and critic agents with different roles
- **Critique Feedback Improvements**: Critique feedback improves the performance of the agents
- **Memory Augmentation**: Stores and utilizes past interactions for better performance
- **Performance Metrics**: Comprehensive evaluation using ROUGE, BERTScore, and novelty metrics
- **Statistical Analysis**: Detailed analysis of agent performance improvements

## Setup

1. Clone the repository:
```bash
git clone git@github.com:rathorevedant99/nlp-polymind.git
cd nlp-polymind
```

2. Create and activate virtual environment:
```bash
python3 -m venv .env
source .env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install hydra-core  # For configuration management
```

## Usage

### Basic Usage

Run the project with default configuration:
```bash
python main.py
```

### Experimentation

To run the experiments, move the experiment file to the root directory and run it.


### Configuration Options

The project uses Hydra for configuration management. All options can be modified through the `configs/config.yaml` file or via command line arguments.

#### Mode
- `mode`: Set to `prod` for production, `dev` for testing with smaller datasets

#### Critic Configuration
- `critic.type`: Model type (`causal` or `seq2seq`)
- `critic.name`: Model name (e.g., `google/gemma-3-4b-it`)
- `critic.device`: Device to run on (`cuda` or `cpu`)
- `critic.is_openrouter`: Whether to use OpenRouter API

#### Expert Configuration
- `experts.num_experts`: Number of expert agents (default: 2)
- `experts.type`: Model type (`causal` or `seq2seq`)
- `experts.name`: Base model name (e.g., `google/flan-t5-small`)
- `experts.device`: Device to run on (`cuda` or `cpu`)
- `experts.debate_rounds`: Number of debate rounds (default: 5)
- `experts.feedback_size`: Size of feedback history (default: 100)
- `experts.batch_size`: Batch size for processing (default: 10)

#### Model Parameters
- `model_params.max_new_tokens`: Maximum tokens to generate (default: 256)
- `model_params.temperature`: Sampling temperature (default: 0.7)
- `model_params.do_sample`: Whether to use sampling (default: true)
- `model_params.top_p`: Top-p sampling parameter (default: 0.9)
- `model_params.num_return_sequences`: Number of sequences to return (default: 1)
- `model_params.min_new_tokens`: Minimum tokens to generate (default: 10)

#### LoRA Configuration
- `lora.enabled`: Whether to use LoRA fine-tuning (default: true)
- `lora.r`: LoRA rank (default: 8)
- `lora.lora_alpha`: LoRA alpha parameter (default: 16)
- `lora.lora_dropout`: LoRA dropout rate (default: 0.1)
- `lora.bias`: Bias type (`none` or other options)

#### Training Configuration
- `training.output_dir`: Directory for model outputs (default: `./results`)
- `training.eval_strategy`: Evaluation strategy (`steps` or other options)
- `training.save_strategy`: Model saving strategy
- `training.per_device_train_batch_size`: Training batch size per device
- `training.per_device_eval_batch_size`: Evaluation batch size per device
- `training.gradient_accumulation_steps`: Steps for gradient accumulation
- `training.eval_accumulation_steps`: Steps for evaluation accumulation
- `training.eval_steps`: Steps between evaluations
- `training.max_steps`: Maximum training steps
- `training.logging_steps`: Steps between logging
- `training.learning_rate`: Learning rate (default: 1e-5)
- `training.weight_decay`: Weight decay (default: 0.01)

#### Data Configuration
- `data.data_cache_dir`: Directory for data caching (default: `./data/cache`)
- `data.category`: Task category (`summarization`, `math`, or `translation`)
- `data.name`: Dataset name (`samsum`, `gsm8k`, or `opus`)
- `data.split`: Data split configuration

### Example Configurations

Basic summarization setup:
```yaml
mode: prod
critic:
  type: causal
  name: google/gemma-3-4b-it
experts:
  num_experts: 3
  type: seq2seq
  name: google/flan-t5-small
data:
  category: summarization
  name: samsum
```
## Experiments

The project includes several experimental scripts:
- `experiments/memory-size-ablation.py`: Memory size impact analysis
- `experiments/num-experts-ablation.py`: Expert count analysis

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the [SPIN Paper](https://arxiv.org/abs/2401.01335)
- Inspired by [SPIN GitHub Repo](https://github.com/uclaml/SPIN)
