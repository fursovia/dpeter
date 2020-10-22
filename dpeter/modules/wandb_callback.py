from typing import Dict, Any

from allennlp.training import EpochCallback, GradientDescentTrainer
import wandb


@EpochCallback.register("wandb")
class WnBCallback(EpochCallback):

    metrics_to_include = [
        "training_loss",
        "training_reg_loss",
        "validation_cer",
        "validation_wer",
        "validation_acc",
        "validation_loss",
    ]

    def __call__(
        self,
        trainer: GradientDescentTrainer,
        metrics: Dict[str, Any],
        epoch: int,
        is_master: bool,
    ) -> None:
        wandb.log({key: val for key, val in metrics.items() if key in self.metrics_to_include})
