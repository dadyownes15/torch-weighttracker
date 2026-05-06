# Lets define a tracker class , which ahs required calculations, and a calc, and a format, it prints out. This is class is specifically for getting metrics, in a nice and readible manner right, but should also be able to be used as a raw data collector. I am also thinking that the format should be easilly intergratable with WAND ina sense, that the format should return a way the renders it nice on that. 

# Also the units it may use, often has the grad avaialbe, because for example they overlap with computing a regularizer for isntance. However, we dont want the it present on the tracker right?

# Perhabs the avbstract is a little bit different: We can sub trackers and main trackers. When people return a tracker, the different metrics they want to track should be combined right? or what do you think? 

# We also need a enum of avaiable trackers, and their required calculations, for the the structtraxcker.
