def load(file_name):
    """
    This function will load a file (variable, classe object, etc) in pickle
    format in to its python orinal variable format.\n
    **Parameters**:\n
    file_name: string type, containg the name of the file to load.\n
    directory_path= is the adreess of the folder where the desired file is
    located.
    **Returns**\n

    Variable: could be an python variable, class object, list, ndarray
    contained in the binary file. \n

    **Example** \n
    import high_order_decomposition_method_functions as hf  \n

    FF=hf.load('example_file')

    """

    if type(file_name) != str:
        file_name_error="""
        The parameter file_name must be a string type variable
        """
        raise TypeError(file_name_error)
    file_in=open(file_name,'rb')
    Object=pickle.load(file_in)
    file_in.close()
    return Object
#-----------------------------------------------------------------------------
def save(variable, file_name):
    """
        This function will save a python variable (list, ndarray, classe
    object, etc)  in a pickle file format .\n
        **Parameters**:\n
            Variable= list, ndarray, class object etc.\n
            file_name= String type. Name of the file that is going to be
            storage. \n
            directory_path=string type. Is the directory adress if the file
            is going to be saved in a desired folder.
        **Returns**:\n
            File= Binary file that will reproduce the object class when
            reloaded. \n
    **Example**\n
    import high_order_decomposition_method_functions as hf

    hf.save(F,'example_file') \n

    Binary file saved as'example_file'
    """


    if type(file_name)!=str:
        raise ValueError('Variable file_name must be a string')
    pickle_out=open(file_name,"wb")
    pickle.dump(variable, pickle_out)
    pickle_out.close()
    print('Binary file saved as'+"'"+file_name+"'")
