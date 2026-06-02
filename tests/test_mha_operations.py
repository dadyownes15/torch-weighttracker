import torch

from torch_weighttracker.operations import WeightOperationType, mha


def test_fused_qkv_source_operation_returns_reduced_source_rows():
    operation = mha.QKVSourceOperation(operation_type=WeightOperationType.SUM)
    weight = torch.arange(1.0, 49.0).reshape(12, 4)

    result = operation(weight)

    torch.testing.assert_close(result, weight.sum(dim=1))


def test_separate_qkv_source_operation_returns_flat_qkv_sources():
    operation = mha.QKVSourceOperation(operation_type=WeightOperationType.SUM)
    q_weight = torch.arange(1.0, 17.0).reshape(4, 4)
    k_weight = torch.arange(17.0, 33.0).reshape(4, 4)
    v_weight = torch.arange(33.0, 49.0).reshape(4, 4)

    result = operation((q_weight, k_weight, v_weight))

    expected = torch.cat(
        [
            q_weight.sum(dim=1),
            k_weight.sum(dim=1),
            v_weight.sum(dim=1),
        ]
    )
    torch.testing.assert_close(result, expected)


def test_fused_qkv_embed_dim_sum_reduces_qkv_rows_in_one_operation():
    operation = mha.FusedQKVEmbedDimOperation(
        operation_type=WeightOperationType.SUM,
        embed_dim=4,
    )
    weight = torch.arange(1.0, 49.0).reshape(12, 4)

    result = operation(weight)

    row_sums = weight.sum(dim=1).reshape(3, 4)
    expected = row_sums.sum(dim=0)
    torch.testing.assert_close(result, expected)


def test_fused_qkv_head_sum_reduces_heads_in_one_operation():
    operation = mha.FusedQKVHeadOperation(
        operation_type=WeightOperationType.SUM,
        num_heads=2,
        head_dim=2,
    )
    weight = torch.arange(1.0, 49.0).reshape(12, 4)

    result = operation(weight)

    row_sums = weight.sum(dim=1).reshape(3, 2, 2)
    expected = row_sums.sum(dim=(0, 2))
    torch.testing.assert_close(result, expected)


def test_separate_qkv_head_sum_reduces_tuple_in_one_operation():
    operation = mha.SeparateQKVHeadOperation(
        operation_type=WeightOperationType.SUM,
        num_heads=2,
        head_dim=2,
    )
    q_weight = torch.arange(1.0, 17.0).reshape(4, 4)
    k_weight = torch.arange(17.0, 33.0).reshape(4, 4)
    v_weight = torch.arange(33.0, 49.0).reshape(4, 4)

    result = operation((q_weight, k_weight, v_weight))

    row_sums = torch.stack(
        [
            q_weight.sum(dim=1),
            k_weight.sum(dim=1),
            v_weight.sum(dim=1),
        ],
    ).reshape(3, 2, 2)
    expected = row_sums.sum(dim=(0, 2))
    torch.testing.assert_close(result, expected)


def test_multihead_attention_head_squared_sum_includes_qkv_and_out_proj_unions():
    operation = mha.MultiheadAttentionSemanticOperation(
        operation_type=WeightOperationType.SQUARED_SUM,
        embed_dim=4,
        num_heads=2,
        mode="head",
    )
    in_proj_weight = torch.ones(12, 4)
    out_proj_weight = torch.ones(4, 4)

    result = operation((in_proj_weight, out_proj_weight))

    torch.testing.assert_close(result, torch.tensor([48.0, 48.0]))
