# coding=utf-8
import numpy as np
import scipy.optimize as sopt

 #This code solves the following problem:
#    minimize_B   0.5*||X - B*S||^2
#   subject to   ||B(:,j)||_2 <= l2norm, for all j=1...size(S,1)

# from util import *

# Lagrange Dual Function to be maximized w/r/t lambda_vars
# Returns negative value because we want to maximize it using a minimization function

Tr = np.trace
inv = np.linalg.inv

def lagrange_dual_factory(X, S, c_const):
    def mini_me(lambda_vars):
        Lambda = np.diag(lambda_vars)

        return (
                -1 * Tr(X.T @ X)
                - Tr((X @ S.T) @ (inv(S @ S.T + Lambda)) @ (X @ S.T).T)
                - Tr(c_const * Lambda)
        )

    return mini_me

def lagrange_dual_learn(X, S, n, c_const, L_init=None, method='CG'):
    if L_init is None:
        L_init = np.zeros(n)
    # Solve for optimal lambda
    lambda_vars = sopt.minimize(
        lagrange_dual_factory(X, S, c_const),
        L_init,
        method=method
    )

    # Set Lambda
    Lambda = np.diag(lambda_vars.x)

    # Returns B^T, for B corresponding to basis matrix
    B = (np.linalg.inv(S @ S.T + Lambda) @ (X @ S.T).T).T
    return (B)

# #LAZY BOI TESTS
n0 = 60
m0 = 50
k0 = 30

B = lagrange_dual_learn(S = np.random.randint(5, size = (n0,m0)), X = np.random.randint(5, size = (k0,m0)), n = n0, c_const = 0.001)
print(B.shape)
