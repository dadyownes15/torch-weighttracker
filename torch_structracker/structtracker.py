class StructTracker():
    def __init__(self,model
                 ) -> None:
        self.calculations = []
        self.regularizers = []
        self.trackers = []
        # Build depedency groups
        self.groups = []


        
        pass



        """
                # Both the create_reguarlizer and create_trackers need a smart way to reuse calculations, while also being able to create new. This means that we have to thoroughly extend  the intialize_from_groups. However, maybe we should move this into the trackers and regularizers, such that they are responsible for creating the calculations, if they are not already present in the current calculations. This also means that we need to on creation, to update the calculations in the local with the one creatied from intiialize groups. 


                We also need a quite a huge rework for the logic around the intialzie from groups, as this needs to be more generalized, however, createSpec, add reduction, should_processtypes, seems to be functions thawe can reuse. However ereate spec should be remanemd to create Reduction. 

                ALso, we need to make sure that the future has capabilityies of adding more advanced layers etc, by moding unifieds methods. this follows natureally of good software princples. 

        def initialize_from_groups(groups: List[Group]): 
    reductions = {}
    unit_count = 0
    for group in groups:
        for member in group.items:    
            if should_process_type(member.dep.target.type):
                reduction, mapping = createSpec(member,unit_count)
                add_reduction(reduction,mapping,reductions)
            else: 
                continue
        unit_count += len(group[0].root_idxs)    
    return reductions, unit_count
        
        """


    def create_regularizer(self, type,device):
        # Creating regularzers and trackers should spawn calcuations units and save these. Each regularizer & tracker sghould have a set of calculations unit required. Ensure to check the calculations units, before creating new, as the goal is to create

        # Also consider some user friendliness regaridng the device, if model ahs a device forexample, what do we do? do models have devices often? perhabs not, we should make it required, but if it already exist, then maybe not. 
        pass


    def create_trackers(self,type,device):
        pass



    # Adds a calculation to 
    def _add_calculation(self):
        pass






# Define a enum for the different trackers
# trackers

# define a enum for the different regualrizers, we only have grouplasso for now but in the futrue we will have sparsity etc