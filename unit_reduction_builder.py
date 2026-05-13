def compile_unit_reduction_plan(
    groups,
    *,
    operation_type
) -> ReductionPlan:
    builder = ReductionPlanBuilder(output_length=count_group_units(groups))
    group_offset = 0

    for group in groups:
        group_size = len(group[0].root_idxs)

        for member in group:
            op = operation_for_member(member, operation_type)
            reducer = WeightReducer(
                parameter_extractor=ParameterExtractor(member.dep.target.module),
                operation=op,
            )

            destination_indices = tuple(
                group_offset + int(index) for index in member.root_idxs
            )

            builder.add(
                ReductionRecord(
                    op=reducer,
                    mapping=ReductionMapping(
                        source=FullSelection(),
                        target=IndexSelection(destination_indices),
                    ),
                )
            )

        group_offset += group_size

    return builder.finalize()
