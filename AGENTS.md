# Repository Instructions

- When adding, renaming, or changing a tracker, update
  `WeightTracker.create_tracker` with the tracker enum value, accepted string,
  output metrics, and public tracker-specific kwargs.
- Keep `TrackerType`, `tracker_class_for_type`, README examples, and focused
  tracker tests aligned with the registered trackers.
- Shared static tensors or state needed by calculations must be modeled as
  `CalcType` calculations and wired through `required_calculations`, not passed
  as ad-hoc tensor kwargs or private buffers on dependent calculations.
- Composite calculations should accept only their calculation dependencies;
  constructor buffers are for static/reduction calculation primitives, not
  calculation graph wiring.
