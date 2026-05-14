Fixes:


    1. Method for ignoring modules for certain ops
        - Grouplasso, keep it
        - Tracker, ignore it

def best solution: 

Instead of doing this, we will add a module / axis filter, which we can inject as a calc
    1. Avoid maintaining different calcs for different filters
    2. Iniate it appart of the create regularizer or tracker
    3. Reuse calcs
    

Errors inside create_units_to_module_axis_plan()


    target = module_index * 2 + axis: Assumes 2 axis for all modules 


## THe indexiing problem

To be able to make tensor operations, we must ensure that we have a unified index system

as of know we have a weighted_module index, which tracks a index for each module

but we have no for unit axis, instead we have assume that module_index*2 = in_channels
and module_index * 2 + 1 =out_channel.

But this does not really fit the for batch norm for example. 

We also have a problem with not


If we could build something like

module_map:
[module_1,module_2,...]

module_axis_map:
[module_1_in,module_2_out,module_3_feature,]

module_map.get(member.module)
module_map.get(member.module,member.axis)


this also allow us to skip or drop certain modules, or axises in calcs. 

## Layer ignoring


For doing grouplasso, we want to include batchnorm, when we do tracking we dont.

How do we encode such ignore things into the calcs. Two approaches as i see it:

#### Approach 1

Make regularizer and trackers take ignore_module

when doing get_calc, require calcs, that not only match the calc but also the ignore

when building plans, it skips the ignored layer


## Down sides

  1. We cannot reuse calcs that has different ignores
  2. Cannot reference a global list using a calc specific indexing, as we have no way of syncing
  3. we apply the ignore on calcs that might not be effected by it 

#### Approach 2

If we want to ignore batchnorms in structured bobs, we can make a simple fix: skip the batchnorms when doing the plan for the active units. If we do this we get the rest for free.



