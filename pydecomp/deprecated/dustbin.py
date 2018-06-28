#------------------------------------------------------------------------------
def finding_biggest_mode(PHI):
    # @Diego Overspecialized and poorly documented. What does it work on ?
    # Please move it accordingly.
    """
    Returns the index of the maximal value in a list
    """

    PHI_shape=[np.size(i) for i in PHI]
    index, value = max(enumerate(PHI_shape), key=operator.itemgetter(1))

    return index
#------------------------------------------------------------------------------

def rearrange(PHI,index):
    # @Diego Overspecialized and poorly documented. What does it work on ?
    # Please move it accordingly.
    """
    Given a list *PHI* and an *index* this function will return a new list
    with the "index" element of the list as the first element.
    """
    if index > len(PHI):
        raise ValueError("index is higher than number of dimentions");
    PHI.insert(0,PHI.pop(index))
    return PHI

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def new_forme_W(PHI):
    """
    This function returns the shape as list of the core matrix in a tucker
    type decomposition.
    """
    forme_W=[i.shape[0] for i in PHI]

    return forme_W

#------------------------------------------------------------------------------
def final_arrange(W,index):
    """
    Returns an array with the *"index"* dimention as the firs dimention of
    the array.
    """
    W=np.moveaxis(W,0,index)

    return W
