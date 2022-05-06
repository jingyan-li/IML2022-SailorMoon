# Triplet Image Similarity Detection

## Run

`bsub -W 24:00 -J "iml-train-baseline" -N -R "rusage[mem=16384,ngpus_excl_p=1]" -o ../logs/train-baseline.out -n 5 "python -m train --config configs/train.yaml"`

### Experiments

`bsub -W 24:00 -J "iml-train-pretrain" -N -R "rusage[mem=16384,ngpus_excl_p=1]" -o ../logs/train-pretrain.out -n 5 "python -m train --config configs/train_pretrained.yaml"`

`bsub -W 24:00 -J "iml-train-pretrain-lastLayer" -N -R "rusage[mem=16384,ngpus_excl_p=1]" -o ../logs/train-pretrain-lastLayer.out -n 5 "python -m train --config configs/train_pretrained_featureExtract.yaml"`

## Inference

`bsub -W 4:00 -J "iml-infer-baseline" -N -R "rusage[mem=16384,ngpus_excl_p=1]" -o ../logs/infer-baseline.out -n 5 "python -m inference"`