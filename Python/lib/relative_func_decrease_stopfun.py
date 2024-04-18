def relative_func_decrease_stopfun(manopt_problem, x, info, last, rel_func_decrease_tol):
    """
    This function provides an additional stopping criterion for the Manopt
    library: stop whenever the relative decrease in the function value
    between two successive accepted update steps is less than
    rel_func_decrease_tol

    :param manopt_problem: Manopt problem
    :param x: Current iterate
    :param info: Structure containing information about iterations
    :param last: Index of the last iterate
    :param rel_func_decrease_tol: Relative function decrease tolerance
    :return: Boolean indicating whether to stop the optimization
    """

    this_iterate_accepted = info[last]['accepted']
    
    if not this_iterate_accepted or last == 0:
        return False
    else:
        # This iterate was an accepted update step, so get the index of the
        # most recent previously-accepted update
        accepted_array = [info[i]['accepted'] for i in range(last)]
        previous_accepted_iterate_idx = next((i for i in range(last-1, 0, -1) if accepted_array[i]), None)
        
        if previous_accepted_iterate_idx is not None:
            # Get the function value at the previous accepted iterate
            previous_val = info[previous_accepted_iterate_idx]['cost']
            current_val = info[last]['cost']
            
            if (previous_val - current_val) / previous_val < rel_func_decrease_tol:
                return True
            else:
                return False
        else:
            # There have been no previously-accepted update steps
            return False