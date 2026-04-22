# Token-Generation Latency Benchmarking in LLaMA

A benchmarking and analysis project for measuring **token generation latency** in large language models such as **LLaMA** and **TinyLlama**.

This project analyzes how latency changes with **model size**, **prompt length**, **numeric precision**, and **KV-cache usage**, and provides both CLI workflows and a **Streamlit dashboard** for interactive exploration.

---

## Overview

Token generation latency consists of three main components: **TTFT (Time to First Token)** which is the delay before the first token is generated, **per-token latency** which is the time taken to generate each subsequent token, and **end-to-end latency** which is the total time for full generation. This repository helps you measure, compare, and understand these metrics.

---

## Project Goals

- Build a repeatable benchmark harness for TTFT, per-token, and end-to-end latency  
- Break latency into components like attention, MLP, KV-cache, and decoding  
- Study scaling across model sizes, prompt lengths, and precision  
- Identify compute vs memory bottlenecks  
- Evaluate KV-cache optimization strategies  

---

## Repository Structure

- analysis/ – scripts for analyzing benchmark results and generating insights  
- bench/ – core benchmarking logic, model loading, and experiment execution  
- configs/ – configuration files for models, parameters, and experiment settings  
- docs/ – generated reports and markdown summaries of findings  
- optimization/ – KV-cache optimization experiments and proposal implementations  
- plots/ – generated graphs and visualizations (latency, scaling, comparisons)  
- profiling/ – utilities to break down latency into components (attention, MLP, etc.)  
- report/ – assets used for final reports or presentations  
- results/ – output files from experiments (JSON results, CSV summaries)  
- scripts/ – runnable scripts to execute experiments and pipelines  
- streamlit_app.py – interactive dashboard to visualize and compare results  
- requirements.txt – list of required Python dependencies  
- README.md – project documentation and usage guide  

---

## Setup

Create and activate a virtual environment, then install dependencies:

python3.11 -m venv latency-env  
source latency-env/bin/activate
pip install -r requirements.txt  

---

## Running Experiments

You can run different experiments depending on your need:

Run full pipeline: python3.11 scripts/run_full_pipeline.py  

Run single benchmark: python3.11 scripts/run_single.py  

Compare KV-cache vs no cache: python3.11 scripts/run_kv_cache_compare.py  

Run precision × KV-cache experiment:  
python3.11 scripts/run_precision_kv_matrix.py --model meta-llama/Llama-3.2-1B-Instruct --device auto --precisions float32 float16 --max-new-tokens 24 --warmup-runs 2 --measured-trials 3  

---

## Dashboard

Launch the Streamlit dashboard:

python3.11 -m streamlit run streamlit_app.py  

Then open: http://localhost:8501  

---


## Outputs

The project generates JSON benchmark results, CSV summaries, latency plots, scaling graphs, precision comparisons, and KV-cache analysis outputs. These are stored in the results/, plots/, and docs/ folders.

---

## What This Project Analyzes

- TTFT vs model size  
- Per-token latency vs prompt length  
- FP32 vs FP16 trade-offs  
- KV-cache effectiveness  
- Compute vs memory bottlenecks  

---



