# Argumentation Computation with Large Language Models: A Benchmark Study

---

This repository contains code to generate datasets and run experiments for using large language models (LLMs) to compute extensions of various abstract argumentation semantics.

## Installation

---

First, create a Python 3 virtual environment (tested with Python 3.10) and install the required packages using pip or conda.

### Pip

1. **Install PyTorch** (tested with version 2.5.1)

   Make sure to install PyTorch with the CUDA/CPU settings appropriate for your system.

2. **Install other dependencies**

   pip install -r requirements.txt

### AF Generators

Navigate to the `src/data/generators/vendor` directory and compile the Argumentation Framework generators:

   ./install.sh

> *Compiling requires Java, Ant, and Maven.*

## Generate Data

To generate Argumentation Frameworks (AFs), we use:

- [AFBenchGen2](https://sourceforge.net/projects/afbenchgen/)
- [AFGen](http://argumentationcompetition.org/2019/papers/ICCMA19_paper_3.pdf)
- [probo](https://sourceforge.net/projects/probo/)

> Thanks to the original authors!

The `src/data` folder contains data generation scripts:

- `generate_apx.py`: Generates AFs in APX format.
- `apx_to_afs.py`: Converts APX files to `ArgumentationFramework` objects and computes extensions and argument acceptance.
- `afs_to_enforcement.py`: Generates and solves status and extension enforcement problems for an AF.

To generate data, simply run:

   ./generate_data.sh

## Experiments

1. **Download** [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and place it as the `llama_factory` folder.
2. Enter the `llama_factory` directory and install the requirements following the `README.md`.

3. **Download models:**
   - [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
   - [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

4. Copy your train and test datasets into `llama_factory/data`, and update the dataset information accordingly.

5. Go to `examples/train_lora`, edit `llama3_lora_sft.yaml`, then start training:

   llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml

---

## Reference

This project builds on code and ideas from the following:

1. Craandijk, Dennis, and Floris Bex. "Enforcement heuristics for argumentation with deep reinforcement learning." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 36. No. 5. 2022.

2. Zheng, Yaowei, et al. "Llamafactory: Unified efficient fine-tuning of 100+ language models." *arXiv preprint arXiv:2403.13372* (2024).