
from torch_structracker.calculations.base import CalculationType
from torch_structracker.trackers.base import BaseTracker


class StructuredBOPsTracker(BaseTracker):
    required_calculations = (CalculationType.ACTIVE_UNITS,CalculationType.UNITS_TO_MODULE, CalculationType.BITRATE_PR_MODULE, )

    def __init__(self, calculations):
        self.calculations = calculations

    def compute(self):
        # Calculations: active units -> R^(num_units)
        # Calculations: activeUnitsPrModule -> R(num_units) -> R(num_modules)
        # Calculations: Bit_rate_pr_module units -> R^(num_modules)

        # TODO:
        # fix this to be a loop perhabs, and use the calc namings perhabs
        return  self.calculations[2](self.calculations[1](self.calculations[0]))
    
    def toMetric(self, result):
        # TODO:
        # This is simply a formatter of the output from the compute, we should not store the results in the class, very important.
        pass

    def track(self):
        # TODO: Should simply comute, and call the to metric on it
        pass
    

