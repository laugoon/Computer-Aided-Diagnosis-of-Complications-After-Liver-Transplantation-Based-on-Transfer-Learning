"""
L1-penalized minimization using the feature sign search algorithm.
用于求解稀疏编码问题的特征符号搜索算法

minimize_s 0.5*(||xi-B*si||^2 + alpha*si'si + 2*alpha*si(sigma(Lijsj))
%   + mu*si'si + 2*mu*si(sigma(Mijsj)) + gamma*||si||_1)
% X : data matrix
% B : basis matrix
% Sinit : initial coefficient matrix

"""

import logging
import numpy as np
log = logging.getLogger("feature_sign_search_algo")
log.setLevel(logging.INFO)

# 定义函数
def feature_sign_search(dictionary, signal, sparsity, solution=None):
    """
    Solve an L1-penalized minimization problem with the feature
    sign search algorithm of Lee et al (2006).
    Parameters
    ----------
    dictionary : array_like, 2-dimensional   #A
        The dictionary of basis functions from which to form the
        sparse linear combination.
    signal : array_like, 1-dimensional          #y
        The signal being decomposed as a sparse linear combination
        of the columns of the dictionary.
    sparsity : float                          #gamma
        The coefficient on the L1 penalty term of the cost function.
    solution : ndarray, 1-dimensional, optional    #x
        Pre-allocated vector to use to store the solution.
    Returns
    -------
    solution : ndarray, 1-dimensional
        Vector containing the solution. If an array was passed in
        as the argument `solution`, it will be updated in place
        and the same object will be returned.

    """
    effective_zero = 1e-18
    # precompute matrices for speed.
    gram_matrix        = np.dot(dictionary, dictionary.T)
    target_correlation = np.dot(dictionary, signal.T)

    # initialization goes here.
    if solution is None:
        solution = np.zeros(gram_matrix.shape[0])
    else:
        assert solution.ndim == 1, "solution must be 1-dimensional"
        assert solution.shape[0] == dictionary.shape[1], (
            "solution.shape[0] does not match dictionary.shape[1]")

        # Initialize all elements to be zero.
        solution[...] = 0.

    signs      = np.zeros(gram_matrix.shape[0], dtype=np.int8)
    active_set = set()
    z_opt      = np.inf  #无穷大

    # Used to store max(abs(grad[nzidx] + sparsity * signs[nzidx])).
    # Set to 0 here to trigger a new feature activation on first iteration.
    nz_opt = 0

    # second term is zero on initialization.
    grad = -2 * target_correlation + 2 * np.dot(gram_matrix, solution.T)

    max_grad_zerorad_zero = np.argmax(np.abs(grad))
    #返回绝对值最大的索引

    # Just used to compute exact cost function.
    sds = np.dot(signal.T, signal)

    while z_opt > sparsity or not np.allclose(nz_opt, 0):
        if np.allclose(nz_opt, 0):
            candidate = np.argmax(np.abs(grad) * (signs == 0))
            print("cand_feature: %d" % candidate)

            if grad[candidate]> sparsity:
                signs[candidate] = -1.
                solution[candidate] = 0.
                print("add_feature %d with neg sign" % candidate)
                active_set.add(candidate)
            elif grad[candidate] < -sparsity:
                signs[candidate] = 1.
                solution[candidate] = 0.
                print("add_feature %d with pos sign" % candidate)
                active_set.add(candidate)
            if len(active_set) == 0:
                break
        else:
            log.debug("Non-zero coefficient optimality not satisfied, "
                      "skipping new feature activation")
          
        indices    = np.array(sorted(active_set))

        restr_gram = gram_matrix[np.ix_(indices, indices)]

        restr_corr = target_correlation[indices]

        restr_sign = signs[indices]

        rhs = restr_corr - sparsity * restr_sign / 2

        new_solution = np.linalg.solve(np.atleast_2d(restr_gram), rhs)

        new_signs    = np.sign(new_solution)
        restr_oldsol = solution[indices]
        sign_flips   = np.where(abs(new_signs - restr_sign) > 1)[0]

        if len(sign_flips) > 0:
            best_obj = np.inf
            best_curr = None
            best_curr = new_solution
            best_obj = (
                sds + (np.dot(new_solution, np.dot(restr_gram, new_solution)) -
                       2 * np.dot(new_solution, restr_corr)) +
                sparsity * abs(new_solution).sum())
            if log.isEnabledFor(logging.DEBUG):

                # Extra computation only done if the log-level is
                # sufficient.
                ocost = (
                    sds +
                    (np.dot(restr_oldsol, np.dot(restr_gram, restr_oldsol)) -
                     2 * np.dot(restr_oldsol, restr_corr)) +
                    sparsity * abs(restr_oldsol).sum())
                cost = (sds + np.dot(new_solution,
                                     np.dot(restr_gram, new_solution)) -
                        2 * np.dot(new_solution, restr_corr) +
                        sparsity * abs(new_solution).sum())
                # print("Cost before linesearch (old)\t: %e" % ocost)
                # print("Cost before linesearch (new)\t: %e" % cost)
            else:
                ocost = None

            for idx in sign_flips:
                a = new_solution[idx]
                b = restr_oldsol[idx]
                prop = b / (b - a)
                curr = restr_oldsol - prop * (restr_oldsol - new_solution)
                cost = sds + (np.dot(curr, np.dot(restr_gram, curr)) -
                              2 * np.dot(curr, restr_corr) +
                              sparsity * abs(curr).sum())

                if cost < best_obj:
                    best_obj = cost
                    best_prop = prop
                    best_curr = curr

            if ocost is not None:
                if ocost < best_obj and not np.allclose(ocost, best_obj):
                    print("Warning: objective decreased from %e to %e" %
                           (ocost, best_obj))
        else:
            print("No sign flips, not doing line search")
            best_curr = new_solution

        solution[indices] = best_curr
        zeros = indices[np.abs(solution[indices]) < effective_zero]

        solution[zeros]   = 0.
        signs[indices]    = np.int8(np.sign(solution[indices]))

        active_set.difference_update(zeros)
        grad = -2 * target_correlation + 2 * np.dot(gram_matrix, solution)

        # print(grad[signs == 0])
        if len(grad[signs == 0]) == 0:
            break
        z_opt = np.max(abs(grad[signs == 0]))
        nz_opt = np.max(abs(grad[signs != 0] + sparsity * signs[signs != 0]))
    return solution

def learn_coding(dic, sigMat, spar):
    print(dic.shape)
    print(sigMat.shape)
    S = []
    for i in range(sigMat.shape[0]):
        sig = sigMat[i, :]
        Sout = feature_sign_search(dic, sig, spar)
        S.append(Sout)

    S = np.stack(S, axis=1)
    return S





