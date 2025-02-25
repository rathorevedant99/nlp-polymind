## Download the model and run test
python -m mlx_lm.generate \
    --model microsoft/Phi-3.5-mini-instruct \
    --prompt "Who was the first president?" \
    --max-tokens 4096

## Fine-Tune
python -m mlx_lm.lora \
    --model microsoft/Phi-3.5-mini-instruct \
    --train \
    --data ./data \
    --iters 100
