# dpeter

```bash
poetry install
poetry shell
exit
```

## TF

```bash
create_dataset --data-dir ./data
CUDA_VISIBLE_DEVICES="2" tf_train ./data
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