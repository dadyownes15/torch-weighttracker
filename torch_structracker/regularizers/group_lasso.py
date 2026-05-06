from torch_structracker.calculations import CalculationType
from torch_structracker.regularizers.base import BaseRegularizer, RegularizerType


class GroupLasso(BaseRegularizer):
    regularizer_type = RegularizerType.GROUP_LASSO
    required_calculations = (CalculationType.STRUCTURED_UNIT_SUM,)

    def forward(self):
        raise NotImplementedError("GroupLasso is not implemented yet.")
