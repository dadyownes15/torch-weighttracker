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


Defining the active unit paramCount

Calculations: Count params pr unit -> gives a count pr unit 


