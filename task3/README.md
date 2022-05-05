# Triplet Image Similarity Detection

## Run

`bsub -W 24:00 -J "iml-train-baseline" -N -R "rusage[mem=16384,ngpus_excl_p=1]" -o ../logs/train-baseline.out -n 5 "python -m train"`

## Inference

`python -m inference`