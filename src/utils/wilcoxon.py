from scipy.stats import wilcoxon

def wilcoxon_signed_rank(
        method1_mae: list,
        method2_mae: list,
        method1_mse: list,
        method2_mse: list,
        method1_rmse: list,
        method2_rmse: list
) :
    _, p_wilc_mae = wilcoxon(method1_mae, method1_mae, alternative='greater', zero_method='wilcox')        
    _,  p_wilc_mse = wilcoxon(method1_mse, method2_mse, alternative='greater', zero_method='wilcox')        
    _, p_wilc_rmse = wilcoxon(method1_rmse, method2_rmse, alternative='greater', zero_method='wilcox')        
    
    return p_wilc_mae, p_wilc_mse, p_wilc_rmse

    