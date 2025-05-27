## Datasets and Checkpoints

Download the datasets and model checkpoints from the anonymous Hugging Face repository:
**[https://huggingface.co/Anon-SmolTolk](https://huggingface.co/Anon-SmolTolk)**

## Conda Environment

Create the Python environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

Additionaly, you will need to install `fairseq` to generate audio samples.

## Quick Training Run (Default Setup)

You can launch a training run using one of the predefined configuration files under the `config/` directory. For example:

```bash
python train.py config/smoltolk1.7b_stg1.py
```

To override any argument (e.g., disable compilation):

```bash
python train.py config/smoltolk1.7b_stg1.py --compile=False
```

The training script expects the `.bin` files to be located in a folder named after the `feats` argument inside a folder named after the `datasets` argument. For instance, with the configuration in `config/smoltolk1.7b_stg1.py`, the directory structure should be:

```
data/
└── ls960/
    └── mhubert25hzl11/
        ├── train.bin
        ├── train.len
        ├── val.bin
        ├── val.len
        └── meta.pkl
```

If training on multiple datasets (e.g., `datasets = ['ls960', 'libriheavy']`), use the following structure:

```
data/
├── ls960/
│   └── mhubert25hzl11/
│       ├── train.bin
│       ├── train.len
│       ├── val.bin
│       ├── val.len
│       └── meta.pkl
└── libriheavy/
    └── mhubert25hzl11/
        ├── train.bin
        ├── train.len
        ├── val.bin
        ├── val.len
        └── meta.pkl
```

Checkpoints will be saved in the `out/` directory under:
`out/<model_name>/<checkpoint_id>/`

## Inference

An inference notebook is available at:
`notebooks/inference.ipynb`

Before running inference, download the model checkpoint from the Hugging Face repository and place it under the `out/` directory.

## Evaluation

To evaluate a checkpoint on **sBLIMP**, use the `scripts/nll_eval.py` script. First, set up the [`zrc-toolbox`](https://zerospeech.com/toolbox/) and download the `sLM21-dataset`. You will also need HuBERT pre-extracted tokens (available from the Hugging Face repository).

For **cross-modal evaluation** (sStoryCloze and tStoryCloze), use `scripts/cross-modal_eval.py`. Download the cross-modal dataset and place it under the `data/` directory.

### Example: sBLIMP Evaluation

```bash
PYTHONPATH=. python scripts/nll_eval.py \
  --eval_dataset=syntactic \
  --model_name="Anon-SmolTolk/SmolTolk-2B" \
  --checkpoint_id="model_ckpt" \
  --batch_size=8
```

### Example: Cross-modal Evaluation

```bash
PYTHONPATH=. python scripts/cross-modal_eval.py \
  --eval_dataset=scloze_multimodal \
  --model_name="Anon-SmolTolk/SmolTolk-2B" \
  --checkpoint_id="model_ckpt"
```
