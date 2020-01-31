import imli


model = imli.imli(solver="maxhs")
X, y, features = model.discretize("../benchmarks/iris_bintarget.csv")
model.fit(X, y)
print(model.getRule(features))
