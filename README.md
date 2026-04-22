# Token-Generation Latency Benchmarking in LLaMA

A benchmarking and analysis project for measuring **token generation latency** in LLMs such as **LLaMA** and **TinyLlama**.  
This repo studies how latency changes with **model size**, **prompt length**, **precision**, and **KV-cache usage**, and provides both command-line workflows and a **Streamlit dashboard** for interactive comparison. 

---

## Project Goal

The project is organized around five benchmark goals:

1. **Benchmark harness design**  
   Measure:
   - **TTFT (Time to First Token)**
   - **Steady-state per-token latency**
   - **End-to-end latency**

   using warmup runs, repeated trials, and outlier handling.

2. **Latency decomposition**  
   Break token latency into major components such as:
   - embedding
   - attention
   - MLP / feed-forward
   - KV-cache transfer estimate
   - sampling / decoding
   - framework overhead

3. **Scaling analysis**  
   Study how latency changes across:
   - multiple model sizes
   - different prompt lengths
   - different numeric precisions

4. **Architectural bottleneck analysis**  
   Connect measured latency growth to decoder-time compute and memory behavior.

5. **Optimization proposal**  
   Evaluate KV-cache related optimization ideas and estimate their effect on inference latency.

---

## Repository Structure


TokenLatency/
- analysis/         -  analysis scripts and report generation
- bench/            -  benchmark harness, configs, and model registry
- configs/          -  model/config files
- docs/             -  generated findings and markdown reports
- optimization/     -  KV-cache optimization writeups and proposal code
- plots/            -  generated plots and visual summaries
-  profiling/        -  breakdown data generation utilities
-  report/           -  report-related assets
-  results/          -  benchmark JSON outputs and CSV summaries
-  scripts/          -  runnable experiment pipelines
-  streamlit_app.py  -  interactive dashboard
-  requirements.txt
-  README.md

---

## Setup

### Create virtual environment
python3.11 -m venv latency-env

### Activate environment

Mac/Linux:
source latency-env/bin/activate


### Install dependencies
pip install -r requirements.txt

---

## Run Experiments

### Run full pipeline
python3.11 scripts/run_full_pipeline.py

### Run single benchmark
python3.11 scripts/run_single.py

### Compare KV cache vs no cache
python3.11 scripts/run_kv_cache_compare.py

### Precision × KV cache experiment
python3.11 scripts/run_precision_kv_matrix.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --device auto \
  --precisions float32 float16 \
  --max-new-tokens 24 \
  --warmup-runs 1 \
  --measured-trials 3

---

## Run Streamlit Dashboard

python3.11 -m streamlit run streamlit_app.py

Open in browser:
http://localhost:8501






