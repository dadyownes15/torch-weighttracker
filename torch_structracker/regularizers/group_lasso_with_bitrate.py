from torch_structracker.calculations import CalcType
from torch_structracker.regularizers.base import BaseRegularizer, RegularizerType


class GroupLassoWithBitrate(BaseRegularizer):
    regularizer_type = RegularizerType.GROUP_LASSO_WITH_BITRATE

    def forward(self):
        raise NotImplementedError("GroupLassoWithBitrate is not implemented yet.")
