import os
from contextlib import nullcontext
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
from model import load_ckpt

# -----------------------------------------------------------------------------
# I/O
out_dir = "out"
# data
eval_dataset = 'syntactic'
eval_split = 'test'
feats = 'mhubert25hzl11'
submission_path = "submission"
batch_size = 32
# system
device = "cuda"
dtype = "bfloat16"
model_name = "similar_l24_h16_d1024_i4.8k"
checkpoint_id = "iter5000_final"
manifest_path = "manifest.csv"
num_workers = 16
n_codebooks = 1
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging

if "storycloze" in eval_dataset:
    manifest_path = "manifests/sSC/manifest_eval.csv"
elif "topiccloze" in eval_dataset or "tstorycloze_repro" in eval_dataset:
    manifest_path = "manifests/tSC/manifest_eval.csv"
elif eval_dataset == "swuggy":
    manifest_path = "manifests/lexical/gold_wav.csv"
elif "syntactic" in eval_dataset:
    manifest_path = "manifests/syntactic/test.csv"
print(manifest_path)
# -----------------------------------------------------------------------------
out_path = os.path.join(out_dir, model_name, "evals", checkpoint_id)
os.makedirs(out_path, exist_ok=True)

torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

eval_data_dir = os.path.join('data', eval_dataset, feats)
eval_data = np.memmap(os.path.join(eval_data_dir, f'{eval_split}.bin'), dtype=np.uint16, mode='r')
eval_lens = np.memmap(os.path.join(eval_data_dir, f'{eval_split}.len'), dtype=np.uint16, mode='r')
eval_data = np.split(eval_data, np.cumsum(eval_lens)[:-1])

class GetLogProbWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self,
                tokens: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        # sanity check
        bsize, ncodebooks, seqlen = tokens.size()
        audio_input_masks = torch.ones((bsize, seqlen), device=device, dtype=bool)
        # assert seqlen <= self.model.params.block_size, f"Sequence beyond maximum length of {self.model.params.block_size}"
        assert ncodebooks == self.model.n_codebooks, "Sequence shape must match the specified number of codebooks"
        # apply codebook pattern
        input_audio_tokens = self.model.apply_delay_pattern(tokens)
        # compute the frame audio embeddings as the sum of codebook embeddings
        h_raw = sum([self.model.audio_embed[k](input_audio_tokens[:, k]) for k in range(ncodebooks)])
        h = self.model.forward_audio_only(h_raw, audio_input_masks, is_output=False) if self.model.audio_in_layers is not None else h_raw
        # obtain contextual embeddings
        ctx_out = self.model.context_model(
            inputs_embeds=h,
            use_cache=False,
            output_hidden_states=self.model.params.layer_wa_audio or self.model.params.layer_selwa_audio
        )
        h_ctx = ctx_out['last_hidden_state']
        # compute loss
        h_audio = h_ctx
        if self.model.params.layer_wa_audio or self.model.params.layer_selwa_audio:
            stacked_ctx_h = torch.stack(ctx_out['hidden_states'])[1:]
            if self.model.params.layer_wa_audio:
                # compute context as weighted average of contextual representations in all layers
                h_audio = (stacked_ctx_h * F.softmax(self.model.layer_weights, dim=0)).sum(0)
            if self.model.params.layer_selwa_audio:
                selected_layer_weights = F.softmax(self.model.layer_selector(h_audio).permute(2, 0, 1).unsqueeze(-1), dim=0)
                selwa_entropy = -(selected_layer_weights * (selected_layer_weights + 1e-12).log()).sum(dim=0).mean()
                self.model.losses['selwa_entropy'] = selwa_entropy
                h_audio = (stacked_ctx_h * selected_layer_weights).sum(0)
        if self.model.params.raw_speech_residual:
            h_audio = h_audio + h_raw
        if self.model.audio_out_layers is not None:
            h_audio = self.model.forward_audio_only(h_audio, audio_input_masks, is_output=True, multimodal_input=False)
        # apply each audio prediction head to obtain logits per codebook
        logits = torch.stack([self.model.audio_unembed[k](h_audio) for k in range(self.model.n_codebooks)], dim=1).float()
        nll_target = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")
        nll_target = nll_target.view(bsize, seqlen)
        nll_target *= mask
        nll_target = nll_target.sum(1) / mask.sum(1)
        return -nll_target  # Return log probabilities

os.makedirs(out_path, exist_ok=True)
out_fname = os.path.join(out_path, f"{eval_dataset}_{eval_split}_logprobs.txt")

if not os.path.exists(out_fname):
    model = load_ckpt(model_name, f'{checkpoint_id}.pt', device)
    model = GetLogProbWrapper(model)

    tsv_lines = open(manifest_path).readlines()[1:]
    if eval_dataset == "swuggy":
        file_ids = [l.split(',')[0] for l in tsv_lines]
    else:
        file_ids = [l.split(',')[1] for l in tsv_lines]

    class EvalDataset(Dataset):
        def __init__(self, samples, sample_ids):
            self.samples = samples
            self.sample_ids = sample_ids

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            sample = torch.from_numpy(self.samples[idx].astype(np.int64))
            # Use <eos> as <bos> to measure over the entire sentence
            return sample.roll(1)[:-1], sample[:-1], self.sample_ids[idx]
        
    def collate_fn(batch):
        samples, targets, sample_ids = zip(*batch)
        padded_samples = pad_sequence(
            samples,
            batch_first=True,
            padding_value=model.model.audio_pad_token
        )
        padded_targets = pad_sequence(
            targets,
            batch_first=True,
            padding_value=model.model.audio_pad_token
        )
        padding_mask = (padded_samples != model.model.audio_pad_token)
        return padded_samples.unsqueeze(1), padded_targets, padding_mask, sample_ids

    dataset = EvalDataset(eval_data, file_ids)

    with open(out_fname, 'w') as f:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers
        )
        pbar = tqdm.tqdm(total=len(eval_data))
        for sample in dataloader:
            sentences, targets, mask, sample_ids = sample
            with ctx:
                log_probs = model(sentences.to(device), targets.to(device), mask=mask.to(device))
            for sample_id, pseudo_prob in zip(sample_ids, log_probs):
                line = f"{sample_id} {pseudo_prob.item()}\n"
                f.write(line)
                pbar.update(1)


# After writing out the .txt file, compute or collect final scores depending on eval_dataset.
scores_json_path = os.path.join(out_path, "scores.json")

# Helper to load or create the JSON.
def load_or_create_scores_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# Helper to save the JSON.
def save_scores_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

if eval_dataset in ["storycloze", "topiccloze", "scloze", "tcloze", "storycloze_repro"]:

    with open(out_fname, "r") as fp:
        lines = fp.read().split("\n")
    pseudo_probs = {
        parts[0]: float(parts[1])
        for line in lines if (parts := line.split()) and len(parts) > 1
    }
    scores = []
    for sample_id in tqdm.tqdm(pseudo_probs.keys()):
        if "_correct" in sample_id:
            idx, fid, _ = sample_id.split('_')
            incorrect_id = f"{int(idx) + 1}_{fid}_incorrect"
            scores.append(pseudo_probs[sample_id] > pseudo_probs[incorrect_id])
    mean_score = float(np.mean(scores))
    print("Mean Cloze Score:", mean_score)
    # Update the JSON
    scores_dict = load_or_create_scores_json(scores_json_path)
    scores_dict[eval_dataset] = mean_score
    save_scores_json(scores_json_path, scores_dict)
elif "syntactic" in eval_dataset:
    import subprocess
    import pandas as pd

    os.makedirs(f"{submission_path}/syntactic", exist_ok=True)
    subprocess.run(["cp", out_fname, f"{submission_path}/syntactic/{eval_split}.txt"], check=True)
    subprocess.run(["zrc", "benchmarks:run", "sLM21", submission_path, "--sets", "test", "--task", "syntactic"], check=True)
    subprocess.run(["rm", f"{submission_path}/syntactic/{eval_split}.txt"], check=True)

    score_csv_path = "submission/scores/score_syntactic_test_by_type.csv"
    if not os.path.exists(score_csv_path):
        print(
            f"Expected {score_csv_path} does not exist.\n"
            f"Please run:\n"
            f"  zrc benchmarks:run sLM21 submission --sets test --task syntactic\n"
            f"after ensuring APP_DIR is set correctly."
        )
    else:
        df = pd.read_csv(score_csv_path)
        mean_score = df["score"].mean()
        print("Mean syntactic score:", mean_score)
        # Update the JSON
        scores_dict = load_or_create_scores_json(scores_json_path)
        scores_dict[eval_dataset] = mean_score
        save_scores_json(scores_json_path, scores_dict)
elif "swuggy" in eval_dataset:
    import subprocess
    import pandas as pd

    os.makedirs(f"{submission_path}/lexical", exist_ok=True)
    subprocess.run(["cp", out_fname, f"{submission_path}/lexical/{eval_split}.txt"], check=True)
    subprocess.run(["zrc", "benchmarks:run", "sLM21", submission_path, "--sets", "test", "--task", "lexical", "--skip-validation"], check=True)
    subprocess.run(["rm", f"{submission_path}/lexical/{eval_split}.txt"], check=True)

    score_csv_path = "submission/scores/score_lexical_test_by_pair.csv"
    if not os.path.exists(score_csv_path):
        print(
            f"Expected {score_csv_path} does not exist.\n"
            f"Please run:\n"
            f"  zrc benchmarks:run sLM21 submission --sets test --task lexical\n"
            f"after ensuring APP_DIR is set correctly."
        )
    else:
        df = pd.read_csv(score_csv_path)
        mean_score = df["score"].mean()
        print("Mean lexical score:", mean_score)
        # Update the JSON
        scores_dict = load_or_create_scores_json(scores_json_path)
        scores_dict[eval_dataset] = mean_score
        save_scores_json(scores_json_path, scores_dict)
else:
    import tqdm
    import numpy as np
    import json

    with open(out_fname, "r") as fp:
        lines = fp.read().split("\n")
    pseudo_probs = {
        parts[0]: float(parts[1])
        for line in lines if (parts := line.split()) and len(parts) > 1
    }
    scores = []
    for sample_id in tqdm.tqdm(pseudo_probs.keys()):
        if "_correct" in sample_id:
            idx, _ = sample_id.split('_')
            incorrect_id = f"{int(idx)}_incorrect"
            scores.append(pseudo_probs[sample_id] > pseudo_probs[incorrect_id])
    mean_score = float(np.mean(scores))
    print(f"Mean {eval_dataset} {eval_split} Score:", mean_score)
    # Update the JSON
    scores_dict = load_or_create_scores_json(scores_json_path)
    scores_dict[eval_dataset+'_'+eval_split] = mean_score
    save_scores_json(scores_json_path, scores_dict)