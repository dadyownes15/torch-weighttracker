def bitwidths_by_name(bits: dict[str, tuple[int, int]]):
    """`bitwidth_fn` with fixed `(activation, weight)` bits per layer, else 32/32."""

    def bitwidth_fn(name, module):
        return bits.get(name, (32, 32))

    return bitwidth_fn
