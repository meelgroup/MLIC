import cplex
from time import time
import pyrulelearn.utils

def _call_cplex(imli, A, y):
    # A = pyrulelearn.utils._add_dummy_columns(A)

    no_features = -1
    no_samples = len(y)
    if(no_samples > 0):
        no_features = len(A[0])
    else:
        print("- error: the dataset is corrupted, does not have sufficient samples")

    if (imli.verbose):
        print("- no of features: ", no_features)
        print("- no of samples : ", no_samples)

    # Establish the Linear Programming Model
    myProblem = cplex.Cplex()

    feature_variable = []
    variable_list = []
    objective_coefficient = []
    variable_count = 0

    for eachLevel in range(imli.numClause):
        for i in range(no_features):
            feature_variable.append(
                "b_" + str(i + 1) + str("_") + str(eachLevel + 1))

    variable_list = variable_list + feature_variable

    slack_variable = []
    for i in range(no_samples):
        slack_variable.append("s_" + str(i + 1))

    variable_list = variable_list + slack_variable

    if (imli.learn_threshold_clause):
        variable_list.append("eta_clause")

    if (imli.learn_threshold_literal):
        # consider different threshold when learning mode is on
        for eachLevel in range(imli.numClause):
            variable_list.append("eta_clit_"+str(eachLevel))

    for i in range(len(y)):
        for eachLevel in range(imli.numClause):
            variable_list.append("ax_" + str(i + 1) +
                                    str("_") + str(eachLevel + 1))

    myProblem.variables.add(names=variable_list)

    # encode the objective function:

    if(imli.verbose):
        print("- weight feature: ", imli.weightFeature)
        print("- weight error:   ", imli.dataFidelity)

    if(imli.iterations == 1 or len(imli._assignList) == 0):  # is called in the first iteration
        for eachLevel in range(imli.numClause):
            for i in range(no_features):
                objective_coefficient.append(imli.weightFeature)
                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)
                myProblem.variables.set_types(
                    variable_count, myProblem.variables.type.continuous)
                myProblem.objective.set_linear(
                    [(variable_count, objective_coefficient[variable_count])])
                variable_count += 1
    else:
        for eachLevel in range(imli.numClause):
            for i in range(no_features):
                if (imli._assignList[eachLevel * no_features + i] > 0):
                    objective_coefficient.append(-imli.weightFeature)
                else:
                    objective_coefficient.append(imli.weightFeature)

                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)
                myProblem.variables.set_types(variable_count, myProblem.variables.type.continuous)
                myProblem.objective.set_linear([(variable_count, objective_coefficient[variable_count])])
                variable_count += 1

    # slack_variable = []
    for i in range(no_samples):
        objective_coefficient.append(imli.dataFidelity)
        myProblem.variables.set_types(
            variable_count, myProblem.variables.type.continuous)
        myProblem.variables.set_lower_bounds(variable_count, 0)
        myProblem.variables.set_upper_bounds(variable_count, 1)
        myProblem.objective.set_linear(
            [(variable_count, objective_coefficient[variable_count])])
        variable_count += 1

    myProblem.objective.set_sense(myProblem.objective.sense.minimize)

    var_eta_clause = -1

    if (imli.learn_threshold_clause):
        myProblem.variables.set_types(
            variable_count, myProblem.variables.type.integer)
        myProblem.variables.set_lower_bounds(variable_count, 0)
        myProblem.variables.set_upper_bounds(variable_count, imli.numClause)
        var_eta_clause = variable_count
        variable_count += 1

    var_eta_literal = [-1 for eachLevel in range(imli.numClause)]
    constraint_count = 0

    if (imli.learn_threshold_literal):

        for eachLevel in range(imli.numClause):
            myProblem.variables.set_types(
                variable_count, myProblem.variables.type.integer)
            myProblem.variables.set_lower_bounds(variable_count, 0)
            myProblem.variables.set_upper_bounds(variable_count, no_features)
            var_eta_literal[eachLevel] = variable_count
            variable_count += 1

            constraint = []

            for j in range(no_features):
                constraint.append(1)

            constraint.append(-1)

            myProblem.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind=[eachLevel * no_features + j for j in range(no_features)] + [var_eta_literal[eachLevel]],
                                        val=constraint)],
                rhs=[0],
                names=["c" + str(constraint_count)],
                senses=["G"]
            )
            constraint_count += 1

    for i in range(len(y)):
        if (y[i] == 1):

            auxiliary_index = []

            for eachLevel in range(imli.numClause):
                constraint = [int(feature) for feature in A[i]]

                myProblem.variables.set_types(
                    variable_count, myProblem.variables.type.integer)
                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)

                constraint.append(no_features)

                auxiliary_index.append(variable_count)

                if (imli.learn_threshold_literal):

                    constraint.append(-1)

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count,
                                                                                                var_eta_literal[eachLevel]],
                            val=constraint)],
                        rhs=[0],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

                else:

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features +
                                    j for j in range(no_features)] + [variable_count],
                            val=constraint)],
                        rhs=[imli.threshold_literal],
                        names=["c" + str(constraint_count)],
                        senses=["G"]
                    )

                    constraint_count += 1

                variable_count += 1

            if (imli.learn_threshold_clause):

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i + imli.numClause * no_features,
                                var_eta_clause] + auxiliary_index,
                        # 1st slack variable = level * no_features
                        val=[imli.numClause, -1] + [-1 for j in range(imli.numClause)])],
                    rhs=[- imli.numClause],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1

            else:

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        # 1st slack variable = level * no_features
                        ind=[i + imli.numClause * no_features] + auxiliary_index,
                        val=[imli.numClause] + [-1 for j in range(imli.numClause)])],
                    rhs=[- imli.numClause + imli.threshold_clause],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1

        else:

            auxiliary_index = []

            for eachLevel in range(imli.numClause):
                constraint = [int(feature) for feature in A[i]]
                myProblem.variables.set_types(
                    variable_count, myProblem.variables.type.integer)
                myProblem.variables.set_lower_bounds(variable_count, 0)
                myProblem.variables.set_upper_bounds(variable_count, 1)

                constraint.append(- no_features)

                auxiliary_index.append(variable_count)

                if (imli.learn_threshold_literal):

                    constraint.append(-1)

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features + j for j in range(no_features)] + [variable_count,
                                                                                                var_eta_literal[eachLevel]],
                            val=constraint)],
                        rhs=[-1],
                        names=["c" + str(constraint_count)],
                        senses=["L"]
                    )

                    constraint_count += 1
                else:

                    myProblem.linear_constraints.add(
                        lin_expr=[cplex.SparsePair(
                            ind=[eachLevel * no_features +
                                    j for j in range(no_features)] + [variable_count],
                            val=constraint)],
                        rhs=[imli.threshold_literal - 1],
                        names=["c" + str(constraint_count)],
                        senses=["L"]
                    )

                    constraint_count += 1

                variable_count += 1

            if (imli.learn_threshold_clause):

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[i + imli.numClause * no_features,
                                var_eta_clause] + auxiliary_index,
                        # 1st slack variable = level * no_features
                        val=[imli.numClause, 1] + [-1 for j in range(imli.numClause)])],
                    rhs=[1],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1

            else:

                myProblem.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        # 1st slack variable = level * no_features
                        ind=[i + imli.numClause * no_features] + auxiliary_index,
                        val=[imli.numClause] + [-1 for j in range(imli.numClause)])],
                    rhs=[- imli.threshold_clause + 1],
                    names=["c" + str(constraint_count)],
                    senses=["G"]
                )

                constraint_count += 1

    # set parameters
    if(imli.verbose):
        print("- timelimit for solver: ",  imli.timeOut - time() + imli._fit_start_time)
    myProblem.parameters.clocktype.set(1)  # cpu time (exact time)
    myProblem.parameters.timelimit.set(imli.timeOut - time() + imli._fit_start_time)
    myProblem.parameters.workmem.set(imli.memlimit)
    myProblem.set_log_stream(None)
    myProblem.set_error_stream(None)
    myProblem.set_warning_stream(None)
    myProblem.set_results_stream(None)
    # myProblem.parameters.mip.tolerances.mipgap.set(0.2)
    myProblem.parameters.mip.limits.treememory.set(imli.memlimit)
    myProblem.parameters.workdir.set(imli.workDir)
    myProblem.parameters.mip.strategy.file.set(2)
    myProblem.parameters.threads.set(1)

    # Solve the model and print the answer
    start_time = myProblem.get_time()
    start_det_time = myProblem.get_dettime()
    myProblem.solve()
    # solution.get_status() returns an integer code
    status = myProblem.solution.get_status()

    end_det_time = myProblem.get_dettime()

    end_time = myProblem.get_time()
    if (imli.verbose):
        print("- Total solve time (sec.):", end_time - start_time)
        print("- Total solve dettime (sec.):", end_det_time - start_det_time)

        print("- Solution status = ", myProblem.solution.status[status])
        print("- Objective value = ", myProblem.solution.get_objective_value())
        print("- mip relative gap (should be zero):", myProblem.solution.MIP.get_mip_relative_gap())

    #  retrieve solution: do rounding

    imli._assignList = []
    imli._selectedFeatureIndex = []
    # if(imli.verbose):
    #     print(" - selected feature index")
    for i in range(len(feature_variable)):
        if(myProblem.solution.get_values(feature_variable[i]) > 0):
            imli._assignList.append(1)
            imli._selectedFeatureIndex.append(i+1)
        else:
            imli._assignList.append(0)
            # imli._selectedFeatureIndex.append(i+1)
    # print(imli._selectedFeatureIndex)
    
    # imli._assignList.append(myProblem.solution.get_values(feature_variable[i]))

    for i in range(len(slack_variable)):
        imli._assignList.append(myProblem.solution.get_values(slack_variable[i]))

    # update parameters
    if (imli.learn_threshold_clause and imli.learn_threshold_literal):

        imli.threshold_literal_learned = [int(myProblem.solution.get_values(var_eta_literal[eachLevel])) for eachLevel in range(imli.numClause)]
        imli.threshold_clause_learned = int(myProblem.solution.get_values(var_eta_clause))

    elif (imli.learn_threshold_clause):
        imli.threshold_literal_learned = [imli.threshold_literal for eachLevel in range(imli.numClause)]
        imli.threshold_clause_learned = int(myProblem.solution.get_values(var_eta_clause))

    elif (imli.learn_threshold_literal):
        imli.threshold_literal_learned = [int(myProblem.solution.get_values(var_eta_literal[eachLevel])) for eachLevel in range(imli.numClause)]
        imli.threshold_clause_learned = imli.threshold_clause

    if(imli.verbose):
        print("- cplex returned the solution")
