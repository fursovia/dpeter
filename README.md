# dpeter

```bash
poetry install
poetry shell
exit
```

## TF

```bash
create_dataset configs/htr_base.jsonnet --data-dir ./data
CUDA_VISIBLE_DEVICES="5" tf_train configs/finetune.jsonnet ./data
```

## Aling letters
```bash
/home/DeslantImg/DeslantImg input.jpg output.jpg
```

## Torch
```bash
wandb login $YOUR_API_KEY

CUDA_VISIBLE_DEVICES="0" train configs/lc_base.jsonnet
```


## Predict 

```bash
bash bin/create_submit.sh
```



docker run -v ${PWD}:/notebook fursovia/dpeter /bin/bash -c "export PYTHONPATH=. && python dpeter/commands/tf_predict.py ./presets/tf_models/last_submit --data-dir valid_images --out-path valid_out"