from torch_structracker.calculations import CalculationType
from torch_structracker.regularizers.base import BaseRegularizer, RegularizerType


class GroupLasso(BaseRegularizer):
    regularizer_type = RegularizerType.GROUP_LASSO

    def forward(self):
        raise NotImplementedError("GroupLasso is not implemented yet.")
