from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, n_estimators:int, max_depth:int, random_state:int):
  model = RandomForestClassifier(
    n_estimators = n_estimators,
    max_depth=max_depth,
    random_state = random_state
  )
  model.fit(X_train, y_train)
  return model