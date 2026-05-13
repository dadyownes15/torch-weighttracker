
from torch_structracker.calculations import CalcType
from torch_structracker.trackers.base import BaseTracker


class StructuredBOPs(BaseTracker):
    
    required_calculations = (CalcType.UNIT_ACTIVE_MASK,CalcType.UNITS_TO_MODULE_AXIS, CalcType.BITRATE_PR_MODULE) 
a




    