from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
import numpy as np

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / (X.shape[1] * X.var() + 1e-8)
   
    return sk_rbf(X, Y, gamma=gamma)
    
def kmm_weights(X_src, X_tgt, B=2.0, eps=0.1, gamma=None):
    """s
    Kernel Mean Matching: minimize || (1/n_s) sum_i beta_i phi(x_i) - (1/n_t) sum_j phi(z_j) ||^2
    s.t. 0 <= beta_i <= B, | (1/n_s) sum beta_i - 1 | <= eps.
    """
    n_s, n_t = X_src.shape[0], X_tgt.shape[0]
    K_ss = rbf_kernel(X_src, X_src, gamma=gamma)
    K_st = rbf_kernel(X_src, X_tgt, gamma=gamma)
    kappa = (1.0 / n_t) * K_st.sum(axis=1)
    # min  (1/2) beta' Q beta - kappa' beta   with Q = (1/n_s^2) K_ss
    Q = (1.0 / (n_s * n_s)) * K_ss
    try:
        from cvxopt import matrix, solvers
        solvers.options["show_progress"] = False
        P = matrix(2 * Q)
        q = matrix(-kappa.astype(np.float64))
        G = np.vstack([-np.eye(n_s), np.eye(n_s)])
        h = np.hstack([np.zeros(n_s), B * np.ones(n_s)])
        A = np.ones((1, n_s))
        b = np.array([1.0])
        # | (1/n_s) sum beta - 1 | <= eps  =>  two inequalities
        A2 = np.vstack([A / n_s, -A / n_s])
        b2 = np.array([1 + eps, -(1 - eps)])
        G2 = np.vstack([G, A2])
        h2 = np.hstack([h, b2])
        G2 = matrix(G2.astype(np.float64))
        h2 = matrix(h2.astype(np.float64))
        A = matrix(A.astype(np.float64))
        b = matrix(b.astype(np.float64))
        sol = solvers.qp(P, q, G2, h2, A, b)
        if sol["status"] == "optimal":
            beta = np.array(sol["x"]).ravel().astype(np.float32)
            return np.clip(beta, 0, B)
    except Exception:
        pass
    # Fallback: scipy minimize
    from scipy.optimize import minimize
    def obj(b):
        return 0.5 * b @ Q @ b - kappa @ b

    # | (1/n_s) sum beta - 1 | <= eps  =>  (1-eps) <= sum/n_s <= (1+eps)
    cons = [
        {"type": "ineq", "fun": lambda b: (b.sum() / n_s) - (1 - eps)},
        {"type": "ineq", "fun": lambda b: (1 + eps) - (b.sum() / n_s)},
    ]
    res = minimize(obj, np.ones(n_s), method="SLSQP", bounds=[(0, B)] * n_s, constraints=cons)
    if res.success:
        return np.clip(res.x.astype(np.float32), 0, B)
    return np.ones(n_s, dtype=np.float32)
