## Installation

```bash
# Create environment
conda create -n aive python=3.11 -y
conda activate aive

# Install PyTorch (adjust for your CUDA version)
pip install torch>=2.4.0 torchvision>=0.19.0 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

### VLM API keys

```bash
export OPENAI_API_KEY=your_key         # GPT-4o / o4-mini
export GOOGLE_API_KEY=your_key         # Gemini 1.5
export OPENAI_BASE_URL=your_endpoint   # For Azure / vLLM proxies
```

### Local VLMs (optional)

Download InternVL3-8B or Qwen3-VL-8B from HuggingFace and set `model_path` in
`utils/config.py` → `LOCAL_MODEL_CONFIGS`.

## Dataset

The SAT spatial reasoning dataset is downloaded automatically:

```bash
python utils/data_process.py --split val
```

## Running

```bash
# Environment setup
export OPENAI_API_KEY=your_key
export PYTHONPATH=$PYTHONPATH:./

# Run the pipeline
bash scripts/run_aive.sh

# Or directly:
python pipelines/AIVE.py
```

Key arguments (passed via CLI, see `utils/args.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--vlm_qa_model_name` | `gpt-4o` | VLM for question answering |
| `--max_steps_per_question` | `5` | Max exploration steps |
| `--split` | `val` | Dataset split |
| `--output_dir` | `./output` | Results directory |

## Project structure

```
AIVE_code/
├── pipelines/
│   ├── AIVE.py                  # Main pipeline (abstract base)
│   └── AIVE_baseline.py         # Base class (external dependency)
├── utils/
│   ├── config.py                # Model registry & pipeline defaults
│   ├── ModelAdapter.py          # VLM abstraction layer
│   ├── internvl3_adapter.py     # InternVL3-8B local adapter
│   ├── qwen3vl_adapter.py       # Qwen3-VL-8B local adapter
│   ├── metrics.py               # Evaluation & accuracy tracking
│   ├── visualization.py         # Paper figure generation
│   └── prompt.py                # Prompt formatting
├── scripts/
│   └── run_aive.sh              # Shell driver
├── requirements.txt
└── README.md
```
