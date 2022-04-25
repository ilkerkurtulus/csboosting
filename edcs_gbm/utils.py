import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def custom_sigmoid(x, s):
	# for neural networks, this can be useful.
	prod = x*s
	prod = np.where(prod >= 50, 50, (np.where(prod <= -50, -50, prod)))
	k = np.nan_to_num(np.exp(prod))

	return k/(1+k)

def custom_residual(c, cost_alpha, cost_beta, y, p):
	# negative first derivative of the loss function
	return c*(cost_alpha*y + p*((cost_beta - cost_alpha)*y - cost_beta))

def custom_hessian(c, cost_alpha, cost_beta, y, p):
	# second derivative of the loss function
	return -1*c*p*(1-p)*((cost_beta - cost_alpha)*y - cost_beta)

def custom_loss(c, cost_alpha, cost_beta, y, p):
	# cost-sensitive loss function
    p = np.where(p == 0, p + 10**-4, p)
    p = np.where(p == 1, p - 10**-4, p)
    return -1*c*(y*np.log(p)*cost_alpha + (1-y)*np.log(1-p)*cost_beta)

def savings(cost_test, cost_alpha, cost_beta, y, p):
	# cost-sensitve performance metric
    cost_pred = np.sum(custom_loss(cost_test, cost_alpha, cost_beta, y, p))
    cost_0 = np.sum(custom_loss(cost_test, cost_alpha, cost_beta, y, np.zeros(len(y))))
    cost_1 = np.sum(custom_loss(cost_test, cost_alpha, cost_beta, y, np.ones(len(y))))

    return (min(cost_0, cost_1) - cost_pred)/min(cost_0, cost_1)
