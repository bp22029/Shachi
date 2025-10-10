# Exploratory Studies



This document describes advanced research setups enabled by **Shachi**, including:

- Carrying memory to the next life
- Living in multiple worlds
- LoRA weight experiments

<p align="center">
  <img src="exploratory_studies.png" alt="LLM_ABM_Figs Preview" width="1000">
</p>


---

## Carrying Memory to the Next Life 🧬

Shachi allows agents to persist experiences across environments by carrying over their memory.
First, save the agent’s state (e.g., as a `.pkl`) in one environment, then load it into another:

```bash
uv run scripts/main_carrying_memory.py   # replace pkl_path before running
```

This enables agents to transfer prior experiences and demonstrate cross-environment adaptation.

---

## Living in Multiple Worlds 🌍

Agents can also move between multiple tasks in a single continuous run.
For example, starting in a stock trading simulation and then moving to social media:

```bash
uv run scripts/main_stock_oasis.py
```

In this example, the same agent lives in both **OASIS** and **StockAgent**, carrying its knowledge across domains to enable multi-domain interactions.

---

## LoRA Weight Experiments 🎛️

Beyond prompt-based profiling, Shachi supports experiments with **LoRA weight modifications** to study their impact on LLM behavior.
For example, applying **SOTOPIA-π** weights:

```bash
uv run scripts/main_vllm.py --config-name "config_vllm" 'task=psychobench' 'agent=sotopia_vllm' 'launcher/vllm=sotopia_pi'
```





