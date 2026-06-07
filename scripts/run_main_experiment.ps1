param(
    [int]$Epochs = 30,
    [string]$RunTag = "wrr_reproduce"
)

python run_daily_drought_model.py `
    --stations all `
    --seq-len 30 `
    --epochs $Epochs `
    --batch-size 128 `
    --modal-method moving `
    --top-k-per-base 2 `
    --model-dim 96 `
    --run-tag $RunTag
