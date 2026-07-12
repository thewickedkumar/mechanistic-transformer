# mechanistic-transformer

A from-scratch PyTorch implementation of the original Transformer (Vaswani et al., "Attention Is All You Need"), trained as an English → French translator on the [opus_books](https://huggingface.co/datasets/opus_books) dataset.

No `nn.Transformer`, no shortcuts — encoder, decoder, multi-head attention, positional encoding, and layer norm are all built by hand to make the mechanics explicit.

## Files

- `transformer.py` — the model: embeddings, sinusoidal positional encoding, multi-head attention, feed-forward blocks, encoder/decoder stacks, and `build_transformer(...)` to assemble them.
- `dataset.py` — `BilingualDataset`, which tokenizes source/target text and produces padded encoder/decoder inputs, labels, and attention masks (including the causal mask for the decoder).
- `train.py` — tokenizer training/loading, the training loop, greedy-decode validation, TensorBoard logging, and checkpointing.

## Setup

```bash
pip install torch datasets tokenizers tqdm tensorboard
```

## Usage

```bash
python train.py
```

On first run this will:
1. Download the `en-fr` split of `opus_books`.
2. Train (or load, if cached) a `WordLevel` tokenizer per language.
3. Train the transformer, saving a checkpoint to `weights/` after every epoch.
4. Log training loss and a few greedy-decoded validation examples to TensorBoard (`runs/tmodel`).

View training progress with:

```bash
tensorboard --logdir runs
```

## Config

All hyperparameters live in `get_config()` in `train.py`:

| key | default | meaning |
|---|---|---|
| `lang_src` / `lang_tgt` | `en` / `fr` | translation direction |
| `seq_len` | 350 | max token length per sentence |
| `d_model` | 512 | embedding/model dimension |
| `batch_size` | 8 | training batch size |
| `num_epochs` | 20 | training epochs |
| `lr` | 1e-4 | learning rate |
| `preload` | `None` | epoch checkpoint to resume from |

To resume training, set `preload` to an epoch string (e.g. `"05"`) matching a saved file in `weights/`.
