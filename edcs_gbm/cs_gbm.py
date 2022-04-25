import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from .utils import custom_sigmoid, custom_residual, custom_hessian, custom_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class CSGbm(BaseEstimator, ClassifierMixin):


    """
    cost_alpha: cost of FN
    cost_beta: cost of FP

    """

    def __init__(self, n_estimators = 100, max_depth = 8, min_samples_leaf = 5, learning_rate = 0.8, cost_alpha = 0.08, cost_beta = 1, random_state = 42):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.cost_alpha = cost_alpha
        self.cost_beta = cost_beta
        self.random_state = random_state

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, individual_costs):

        # step1: initialize

        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        n = len(X)
        # p0: initial predicted probability
        p = (np.sum(individual_costs*y*self.cost_alpha))/np.sum(individual_costs*(y*(self.cost_alpha - self.cost_beta) + self.cost_beta))

        # first log odds
        f = np.log(p/(1-p))
        self.initial_logodds = f.copy()

        p = np.repeat(p, n)
        f = np.repeat(f, n)

        # step2: iteration

        self.estimators = []
        self.label_encoders = []
        self.output_values = dict()
        self.residuals = []
        self.hessians = []
        self.iteration_loss = []

        for est_idx in range(self.n_estimators):

            in_tree_output_values = dict()

            # j
            res = custom_residual(individual_costs, self.cost_alpha, self.cost_beta, y, p)
            hess = custom_hessian(individual_costs, self.cost_alpha, self.cost_beta, y, p)

            self.residuals.append(res)
            self.hessians.append(hess)

            tmp_clf = DecisionTreeRegressor(max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, random_state = self.random_state)
            tmp_clf.fit(X, res)

            self.estimators.append(tmp_clf)

            # label leaves (RJM)

            le = LabelEncoder()
            leaves = tmp_clf.apply(X)
            le.fit(leaves)
            self.label_encoders.append(le)

            for label_class in le.classes_:
                # m
                leave_idx = np.where(leaves == label_class)
                leave_res = res[leave_idx]
                leave_hess = hess[leave_idx]

                ov_jm = leave_res.sum()/leave_hess.sum()  # output value for the leave to update log(odds) prediction
                in_tree_output_values.update({label_class : ov_jm})

                f[leave_idx] += self.learning_rate*ov_jm

                # calculate new probs

                p[leave_idx] = custom_sigmoid(f[leave_idx], 1)

            # update tree output values

            self.output_values.update({est_idx : in_tree_output_values})

            iter_loss = np.sum(custom_loss(individual_costs, self.cost_alpha, self.cost_beta, y, p))
            self.iteration_loss.append(iter_loss)

        self.train_probs = p

        return self

    def predict(self, X):

        f_initial = self.initial_logodds.copy()
        f_initial = np.repeat(f_initial, len(X))
        output_values = self.output_values

        for est_idx in range(len(self.estimators)):

            le = self.label_encoders[est_idx]
            leaves = self.estimators[est_idx].apply(X)
            le.transform(leaves)

            for label_class in le.classes_:

                leave_idx = np.where(leaves == label_class)

                ov_jm = output_values[est_idx][label_class]
                f_initial[leave_idx] += self.learning_rate*ov_jm

        y_pred = custom_sigmoid(f_initial, 1)

        return np.where(y_pred >= 0.5, 1, 0)


    def predict_proba(self, X):

        f_initial = self.initial_logodds.copy()
        f_initial = np.repeat(f_initial, len(X))
        output_values = self.output_values

        for est_idx in range(len(self.estimators)):

            le = self.label_encoders[est_idx]
            leaves = self.estimators[est_idx].apply(X)
            le.transform(leaves)

            for label_class in le.classes_:

                leave_idx = np.where(leaves == label_class)

                ov_jm = output_values[est_idx][label_class]
                f_initial[leave_idx] += self.learning_rate*ov_jm

        y_pred = custom_sigmoid(f_initial, 1)

        return y_pred
