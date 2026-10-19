import numpy as np
from phase_estimators import KTG, ARTG

rng = np.random.default_rng(42)
X = rng.uniform(-np.pi, np.pi, size=(400, 5))

for Model in (KTG, ARTG):
    estimator = Model(lags=2, group_lasso=True, lambda_=0.04, max_iter=500)
    estimator.fit(X)
    print(Model.__name__, estimator.coef_.shape, estimator.n_iter_,
          estimator.score(X), estimator.edge_strengths().shape)

# Without regularization: MLE by L-BFGS-B
model = KTG(lags=1, group_lasso=False).fit(X)
print('MLE:', model.coef_.shape, model.predict(X).shape)

# Forward-vs-reversed empirical conditional log-likelihood ratio
print('Entropy production estimate (nats/step):', model.entropy_production(X))
