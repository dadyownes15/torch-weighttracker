# View of the architecture



## User facing APIs

StructureAnalyzer
  1. User faced api
  2. Constructs the graph, and compiles module specific information into the graph such as the heads, dimensions, 
  and however one should interprate the graph
  3. Creates Trackers and Regularizers, from the graph. 

Trackers()
  1. Trackers are used for calculating metrics based on the model and its parameterization
  2. Usecases: Structured REL Bobs, deadzone widths etc 

Regularizers()
  1. Used for creating weight based regularizers in a computational efficient manner
  2. Usecases: GroupLasso, Bitrate specific grouplasso etc.


## Some examples of trackers 

The most basic tracker is to sum the total counts of weights in a unit

```python
Tracker(ABC):
  @classMethod
  @torch.no_grad() # very important
  def compute()

  @classMethod
  def toMetric()

  @classMethod
  def track()

  @classMethod
  def validator()
```

```python 
ParamUnitSum(Tracker):
  reguired_calcs = [Calculations.WeightOperation.SUM]
  def __init__(calculation):
    self.calculations = calculation 
  
  def compute() 
    return self.calculations.forward()
    
  def toMetric(input: Calculations.WeightOperation.outputType)
    # tracker based formatting
    return someFormattingCode(input)

 def track() 
    return toMetric(compute())
```

```python
StructuredBobs(Tracker):
  required_calcs = [ 
  Calculation.Mapping.UnitToModule, 
  Calculation.WeightReduction.ActiveUnitParamCount, 
  Calculation.ModuleReduction.BitRatePrModule]

  def compute():
    return tensor.mul(UnitToModule(Calculation.ActiveUnitParamCount),BitRatePrModule)
  
  def toMetric(input: xx )
    return someFormattingCode(input)

  def track():
    return toMetric(compute())
```
## Comments on goal

Goals for this implementiton cycle:
  1. GroupLassoRegularizer
  2. StructuredBobsCalculator (pr module & total)

# Internal APIS

The core architecture design of the structure analyzer is the fact that we have Calculations are nn.Modules
which are initiated before training with the nesacarry data registered on the gpu, to allow fast computation using tensor ops. 
This way, we can minimize training time, by 1) fast comps of regularizers, 2) complex regularizers, which can reduce over multiple aspects
including modules, weights, struftures etc 3) agnostic to any model structure 4) purely device calculations

THe main object which allows this is the Calculation object. 

## CalculationObject

The calculation objects serves to purposes: 
  1. Executing a computional plan and returning a tensor
  2. Creating any registers which needed for the mapping. 
  3. specifying the output shape
  4. Work as the fundament for resuable calculations
  
#### Example

```python
ActiveUnitParamCount(Calculation):
  def __init__(plan: ActiveUnitParamCountPlan)
  ```

  
## ReducerPlans 

There exists two types of reductionPlan 

  1. MappedReductionPlan
  2. ReductionPlan

MappedReductionPlan is defined by a set of reduction operations, where operations must be grouped some kind of mapping. Thus operaitons have a an ouput, which can be mapped to accumulation.

ReductionPlan is defined by the set of reductions operations, where the accumalations of different operations, can some operation R^D -> R^D. 

#### MappedReductionPlan

```python
@dataclass(frozen=True)
class MappedReductionPlan:
    output_length: int
    segment_entries: tuple[SegmentEntry, ...] = ()
    indexed_entries: tuple[IndexedEntry, ...] = ()
    output_labels: tuple[str, ...] | None = None
```

#### ReductionPlan

```python
@dataclass(frozen=True)
class ReductionPlan:
    output_length: int
    operations: tuple[ReductionOp]
    output_labels: tuple[str, ...] | None = None
```

## Entries
For mappedReductionPlan, we cannot simply store the operations, and execute them in a pipeline. Instead we must for each operation
explain how it should map output. It is the mapping, that allows us peform calculations such as active param unit, and many other interesting things. 

We have two types of entries, a segment_entry and index_entry. This is because we have many operations, where the output of the operation
directly maps to the output with an offset, thus leading to a simpler accummalation. For such output we use segmnet entry. For sporadic outputs
we use index_entries.

This can be extended further. 

```python
@dataclass(frozen=True)
class SegmentEntry:
    op: ReductionOp
    start: int
    length: int

@dataclass(frozen=True)
class IndexedEntry:
    op: ReductionOp
    destination_indices: tuple[int, ...]
```

## ReductionOp
ReductionOps Specify the pair of reduction on the source. THe idea is that it has a reference, and know what to do with the reference.
It exposes a shapes, which allows us to confirm if the plan compiles as expected

```python
class TensorReductionOp(nn.Module):
    def __init__(
        self,
        extractor: TensorSourceRef,
        reduction: TensorReduction,
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self.reduction = reduction
        
    @property 
    def validate(self) -> Bool:
        return self.extractor.output_shape == self.reduction.input_shape
        
    @property
    def output_shape(self) -> torch.Size:
        return self.reduction.output_shape

    def identity_key(self) -> Hashable:
        return (
            self.extractor.identity_key(),
            self.reduction.identity_key(),
        )

    def forward(self) -> torch.Tensor:
        return self.reduction(self.extractor.get())

```




# Creating a computation plan 

We know have the recipes, to introduce how computional plans are created. So far we have introduced the following concepts:

1. Calculations
2. ReduceOps 
3. Mapped Reduction Plan


Now we will introduce the concept of a computional plan builder.  The genericReudxcitonPlanner, takes any iterable element, and allows to compiule it into a computional plan, by having a 
reduction rule. hwihc executes for each of the element of the list.

```python
class GenericReductionPlanner(Generic[ElementT]):
    def __init__(
        self,
        elements: Iterable[ElementT],
        *,
        output_length: int | None = None,
    ) -> None:
        self.elements = tuple(elements)
        self.output_length = output_length

    def compile(self, rule: ReductionRule[ElementT]) -> MappedReductionPlan:
        builder = ReductionPlanBuilder(output_length=self.output_length)

        for element in self.elements:
            for record in rule.emit(element, builder):
                builder.add(record)

        return builder.finalize()
```


THe planner essentially, specifies how we compute plans, but requires elements and rules. Rules specificy the details of the reduction, including:

1. The creation of the the reduction operation for the specific element
2. The creation of the target output
3. The creating of the source for the operation if present



```python
class ReductionRule(Protocol[ElementT]):
    def emit(
        self,
        element: ElementT,
        builder: ReductionPlanBuilder,
    ) -> Iterable[ReductionRecord]:
        ...
```


If we want to have a bitrate, we say: 

```python
def create_bitrate_plan(modul: nn.Module):
    planner = GenericReductionPlanner(model.modules())
    plan = planner.compile(
        # the SequentialSegmentMapper just makes the output for each module, map to following indexes
        # Extraxctor gits the bitrate TensorSourceRef
        # We dont need any operations, so it ijsut the identity
        ElementReductionRule[nn.Module](
            targetMapper = SequentialSegmentMapper(),
            extractor = CodeqBitRateExtractor()
            operation = Identity()
        )
        
    )
    return plan 
```

UnitParamSum we can do something like this:

```python
def compile_member_unit_sum_plan(
    groups,
    *,
    operation_type=WeightOperationType.SUM,
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
```

A major update we need to do is ensure that the builder.finalize, has all the logic for compression and compiling the optimial computing. 
We also need to ensure that different moduels can rely on the same informatio nacross calcs

