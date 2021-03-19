def dfs_freeze(model):
    ''' 
    freeze all paramters of a pytorch model
    '''
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)
                                            
