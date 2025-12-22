# ES Translation Fine-tuning

Fine-tune a Seq2Seq translation model using **Evolutionary Strategies (ES)** with **LoRA** on multi-GPU.

## Features

- **ES Optimization**: No gradients needed! Uses population-based perturbation with antithetic sampling
- **LoRA Integration**: Efficient fine-tuning targeting `q_proj`, `k_proj`, `fc2` modules
- **Multi-GPU**: Scales to 8+ GPUs using `torch.distributed`
- **Metrics**: BLEU (active), COMET (ready to use)

## Installation

```bash
pip install -e .
```

## Usage

### Single GPU

```bash
python main.py \
    --model_path /path/to/model \
    --data_path /path/to/data.json \
    --epochs 100 \
    --sigma 0.01 \
    --lr 0.001
```

### Multi-GPU (8 GPUs)

```bash
# Edit paths in run.sh first
bash run.sh
```

Or manually:

```bash
torchrun --nproc_per_node=8 main.py \
    --model_path /path/to/model \
    --data_path /path/to/data.json \
    --population_size 64 \
    --epochs 100
```

## Data Format

JSON file with entries:

```json
[
    {
        "src": "Câu tiếng Việt",
        "prompt": "Câu tiếng Việt",
        "answer": "한국어 문장"
    }
]
```

## Project Structure

```
├── src/
│   ├── models/        # Model loading + LoRA
│   ├── es_engine/     # ES algorithm core
│   ├── data/          # Dataset loading
│   └── utils/         # Metrics (BLEU, COMET)
├── main.py            # Entry point
└── run.sh             # Multi-GPU launch script
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--sigma` | 0.01 | ES noise standard deviation |
| `--lr` | 0.001 | Learning rate |
| `--population_size` | 64 | Number of perturbations |
| `--lora_r` | 8 | LoRA rank |
| `--antithetic` | True | Use mirrored sampling |

## License

GPL-3.0
