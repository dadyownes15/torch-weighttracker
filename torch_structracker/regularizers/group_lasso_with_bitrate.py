from torch_structracker.calculations import CalculationType
from torch_structracker.regularizers.base import BaseRegularizer, RegularizerType


class GroupLassoWithBitrate(BaseRegularizer):
    regularizer_type = RegularizerType.GROUP_LASSO_WITH_BITRATE
    required_calculations = (
        CalculationType.STRUCTURED_UNIT_NORM,
        CalculationType.STRUCTURED_UNIT_COUNT_FROM_NORM,
    )

    def forward(self):
        raise NotImplementedError("GroupLassoWithBitrate is not implemented yet.")
