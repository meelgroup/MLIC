import imli


model = imli.imli(solver="open-wbo", verbose=True)
X, y, features = model.discretize_orange("../benchmarks/iris_orange.csv")
print(features)
model.fit(X, y)
print(model.getRule(features))
print(model)

