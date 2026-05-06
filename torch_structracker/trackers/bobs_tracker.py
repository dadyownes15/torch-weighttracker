from torch_structracker.trackers.base import BaseTracker, TrackerType


class BobsTracker(BaseTracker):
    tracker_type = TrackerType.BOBS_TRACKER
    required_calculations = ()

    def _compute(self, calculations):
        raise NotImplementedError("BobsTracker is not implemented yet.")

    def toMetric(self, result):
        raise NotImplementedError("BobsTracker metrics are not implemented yet.")
