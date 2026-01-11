## checkpoints_best

This directory contains the best-performing model checkpoints for quick inference using `inference.py`.

### Current Models (January 2026 Optimized Version)

| Model | Source | Best Epoch | Valid BLEU | Test BLEU |
|-------|--------|-----------|------------|-----------|
| **RNN** | `checkpoints/checkpoints_rnn/best_rnn_model.pt` | 127/400 | 8.22 | 7.23 |
| **Transformer** | `checkpoints/checkpoints_transformer/best_transformer_model.pt` | 23/100 | 12.60 | 9.34 |
| **mT5** | `checkpoints/checkpoints_t5/best_t5_model.pt` | 18/20 | 15.62 | 10.60 |

### Model Details

- **rnn_best.pt**  
  - Architecture: Unidirectional GRU with General Attention
  - Config: hidden_size=256, embed_dim=256, num_layers=2, dropout=0.3
  - Training: 400 epochs max, early stop at 177, pure Teacher Forcing
  - Valid Loss: 4.6980

- **transformer_best.pt**  
  - Architecture: 4-layer Encoder-Decoder with RMSNorm, Sinusoidal PE
  - Config: d_model=256, num_heads=8, dim_ff=1024, dropout=0.2
  - Training: 100 epochs max, early stop at 43, Noam scheduler
  - Valid Loss: 4.3266

- **t5_best.pt**  
  - Architecture: google/mt5-small (Multilingual T5)
  - Config: 300M parameters, batch_size=16, max_len=128
  - Training: 20 epochs, lr=1e-4, warmup_ratio=0.1
  - Valid Loss: 2.3394

### Usage

```bash
# Demo mode (use all three models)
python inference.py --demo

# Interactive mode
python inference.py --interactive --model_type rnn
python inference.py --interactive --model_type transformer
python inference.py --interactive --model_type t5

# Single sentence translation
python inference.py --model_type rnn --text "今天天气很好"
python inference.py --model_type transformer --text "今天天气很好"
python inference.py --model_type t5 --text "今天天气很好"
```

### Training Logs

- RNN: `outputs/rnn_training_20260110_120533.log`
- Transformer: `outputs/transformer_training_20260110_141214.log`
- mT5: `outputs/t5_training_20260110_161451.log`
