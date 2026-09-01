param(
    [int]$Epochs = 30,
    [int]$ReplicateSeeds = 1,
    [string]$RunTag = "upper_yellow_external_replication"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

python -u scripts/run_upper_yellow_replication.py `
    --stations all `
    --epochs $Epochs `
    --replicate-seeds $ReplicateSeeds `
    --seed 42 `
    --run-tag $RunTag `
    --seq-len 30 `
    --batch-size 128 `
    --modal-method moving `
    --top-k-per-base 2 `
    --meta-trials 1800 `
    --cls-loss-type focal `
    --focal-gamma 1.6 `
    --lambda-cls 1.15 `
    --lambda-idx90 1.25 `
    --lambda-prev-anchor 0.08 `
    --recursive-consistency-weight 0.24 `
    --recursive-prev-blend 0.55 `
    --recursive-unroll-steps 5 `
    --recursive-unroll-decay 0.75 `
    --model-dim 96 `
    --dropout 0.05 `
    --disable-static-dem `
    --lambda-cmip-hist 0 `
    --lambda-cmip-scenario 0 `
    --deep-baseline-epochs 4 `
    --flash-threshold-mode valid_f1 `
    --flash-threshold-objective mcc

exit $LASTEXITCODE
