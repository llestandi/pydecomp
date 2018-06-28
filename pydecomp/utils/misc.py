def list_transpose(phi):
    # @Diego Overspecialized. What does it work on ?
    # Please move it accordingly. If its used in several context
    """
    Parameter:
        phi: list with n-array matrices as elements.

    Return:
        phit: list with n-array transposed matrices of input parameter.
    """
    phit=[i.T for i in phi]
    return phit
