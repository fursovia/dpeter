# dpeter

```bash
poetry install
poetry shell
exit
```

## TF

```bash
create_dataset configs/htr_base.jsonnet --data-dir ./data
CUDA_VISIBLE_DEVICES="3" tf_train configs/htr_base.jsonnet ./data
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

