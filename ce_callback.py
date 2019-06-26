from catalyst.dl.core import MetricCallback
from torch.nn.functional import cross_entropy
from typing import List


class CECallback(MetricCallback):
    """
    Cross entropy metric callback
    """

    def __init__(
            self,
            input_key: str = "targets",
            output_key: str = "logits",
            prefix: str = "cross_entropy",
    ):
        """
        Args:
            input_key: input key to use for accuracy calculation;
                specifies our `y_true`.
            output_key: output key to use for accuracy calculation;
                specifies our `y_pred`.
            accuracy_args: specifies which accuracy@K to log.
                [1] - accuracy
                [1, 3] - accuracy at 1 and 3
                [1, 3, 5] - accuracy at 1, 3 and 5
        """
        super().__init__(
            prefix=prefix,
            metric_fn=cross_entropy,
            input_key=input_key,
            output_key=output_key
        )
