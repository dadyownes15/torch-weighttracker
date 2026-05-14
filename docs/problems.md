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