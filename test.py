
# for testing

import rulelearning
model=rulelearning.imli(verbose=False, solver="maxhs", num_clause=2, data_fidelity=20, rule_type="relaxed_CNF")
X, y, features = model.discretize("benchmarks/iris.csv")
model.fit(X,y)
rule = model.get_rule(features)
print(rule)
print(model.get_selected_column_index())
print(model.get_threshold_clause())
print(model.get_threshold_literal())