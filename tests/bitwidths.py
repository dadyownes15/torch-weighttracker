def bitwidths_by_name(bits: dict[str, tuple[int, int]]) -> dict:
    """WeightTracker kwargs with fixed `(activation, weight)` bits per layer.

    Layers not in `bits` are 32/32.
    """

    def activation_bitwidth_fn(name, module):
        return bits.get(name, (32, 32))[0]

    def weight_bitwidth_fn(name, module):
        return bits.get(name, (32, 32))[1]

    return {
        "activation_bitwidth_fn": activation_bitwidth_fn,
        "weight_bitwidth_fn": weight_bitwidth_fn,
    }
