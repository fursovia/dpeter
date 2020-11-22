# Digital Peter: Recognition of Peter the Great’s manuscripts

[Digital Peter](https://ods.ai/tracks/aij2020/) is an educational task with a historical slant created on the basis
 of several AI technologies (Computer Vision, NLP, and knowledge graphs). 
 The task was prepared jointly with the Saint Petersburg Institute of History 
 (N.P.Lihachov mansion) of Russian Academy of Sciences, Federal Archival 
 Agency of Russia and Russian State Archive of Ancient Acts.


Install requirements

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

