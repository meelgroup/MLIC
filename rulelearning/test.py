import imli


model = imli.imli(solver="open-wbo",  ruleType="checklist", numClause=2)
X, y, features = model.discretize_orange("../benchmarks/iris_orange.csv")
# X, y, features = model.discretize("../benchmarks/iris_bintarget.csv")

model.fit(X, y)
print(model.getRule(features))
print(model)

