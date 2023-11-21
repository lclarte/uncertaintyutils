import numpy as np
import sklearn.linear_model as linear_model
import sklearn.model_selection as model_selection

def get_confidence_interval_jacknife_plus_from_coefs( x_train, y_train, x, coefs_array, coverage):
    """
    coefs_array has shape (n, d)
    """
    n, d = x_train.shape
    alpha = (1.0 - coverage) / 2.0
    train_residuals = np.diag(x_train @ coefs_array.T) - y_train
    lowers, uppers = coefs_array @ x - np.abs(train_residuals), coefs_array @ x + np.abs(train_residuals)

    # NOTE : On divise par n par rapport a https://arxiv.org/pdf/1905.02928.pdf car np.quantile demande une valeur entre 0 et 1
    return np.quantile(lowers,  q=np.floor(alpha * (n+1))/n), np.quantile(uppers, q=np.ceil((1.-alpha) * (n+1))/n)

def get_jacknife_coefs_lasso(x_train, y_train, lambda_, **lasso_args):
    """
    The lambda_ will be rescaled by 1. / (n - 1) so no need to divide it before 
    """
    loo = model_selection.LeaveOneOut()
    n, d = x_train.shape

    coefs_array = np.zeros((n, d))

    for i, (train_indices, _) in enumerate(loo.split(x_train, y_train)):
        x_train_loo, y_train_loo = x_train[train_indices], y_train[train_indices]

        # NOTE : Attention !, diviser par (n-1) permet d'etre coherent avec GAMP car dans sklearn
        # la somme des L2 error est divisee par le nombre de samples (= n - 1 ici) 
        lasso = linear_model.Lasso(alpha = lambda_ / (n - 1), fit_intercept=False, **lasso_args)
        lasso.fit(x_train_loo, y_train_loo)
        coefs_array[i] = lasso.coef_

    return coefs_array
