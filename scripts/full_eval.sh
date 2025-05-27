model_name="ParoleLM/SmolTolk-400M"
checkpoint_id="model_ckpt.pt"

eval_dataset="adel_syntactic_v2"
eval_split="test"
batch_size=8

PYTHONPATH=. python scripts/nll_eval.py --eval_dataset=$eval_dataset --model_name=$model_name --checkpoint_id=$checkpoint_id --batch_size=$batch_size

eval_dataset="storycloze"
eval_split="test"

PYTHONPATH=. python scripts/nll_eval.py --eval_dataset=$eval_dataset --model_name=$model_name --checkpoint_id=$checkpoint_id --batch_size=$batch_size