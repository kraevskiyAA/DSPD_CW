import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler
from scipy import stats


def nlpd_metric(fakes, real):
    """
    The Negative Log Predictive Density (NLPD) metric calculation.
    Source: http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf
    
    Parameters:
    -----------
    fakes : array-like
        List of predictions with shape (n_objects, n_predictions).
    real : array-like
        Real values with shape (n_objects, 1).
        
    Returns:
    --------
    metric : float
        NLPD metrc value.
    """

    fakes_mean = fakes.mean(axis=1)
    fakes_std = fakes.std(axis=1)
    
    metric = (real[:, 0] - fakes_mean)**2 / (2 * fakes_std**2) + np.log(fakes_std) + 0.5 * np.log(2 * np.pi)
    
    return metric.mean()


def nrmse_p_metric(fakes, real):
    """
    The normalized Root Mean Squared Error (nRMSEp) metric based on predicted error.
    Source: http://mlg.eng.cam.ac.uk/pub/pdf/QuiRasSinetal06.pdf

    Parameters:
    -----------
    fakes : array-like
        List of predictions with shape (n_objects, n_predictions).
    real : array-like
        Real values with shape (n_objects, 1).

    Returns:
    --------
    metric : float
        nRMSEp metrc value.
    """

    fakes_mean = fakes.mean(axis=1)
    fakes_std = fakes.std(axis=1)

    metric = (real[:, 0] - fakes_mean) ** 2 / fakes_std ** 2

    return np.sqrt(metric.mean())



def picp_metric(fakes, real, alpha=0.90):
    """
    The Prediction Interval Coverage Probability (PICP) metric. 
    Source: https://www.sciencedirect.com/science/article/pii/S0893608006000153?via%3Dihub
    
    Parameters:
    -----------
    fakes : array-like
        List of predictions with shape (n_objects, n_predictions).
    real : array-like
        Real values with shape (n_objects, 1).
    alpha : float [0, 1]
        Fraction of the distribution inside confident intervals.
        
    Returns:
    --------
    metric : float
        PICP metrc value.
    """

    fakes_mean = fakes.mean(axis=1)
    fakes_std = fakes.std(axis=1)
    
    p_left, p_right = stats.norm.interval(alpha, loc=fakes_mean, scale=fakes_std)
    metric = (real[:, 0] > p_left) * (real[:, 0] <= p_right)
    
    return metric.mean()



def get_quality_metrics(fakes, real):
    """
    Parameters:
    -----------
    fakes : array-like
        List of predictions with shape (n_objects, n_predictions).
    real : array-like
        Real values with shape (n_objects, 1).
        
    Returns
    -------
    List of metric values: [rmse, mae, rse, rae, mape, rmsle, nlpd, nrmse, picp]
    """

    fakes_mean = fakes.mean(axis=1)
    fakes_std = fakes.std(axis=1)
    
    rmse  = np.sqrt( mean_squared_error(real[:, 0], fakes_mean) )
    mae   = mean_absolute_error(real[:, 0], fakes_mean)
    
    rse  = np.sqrt( ( (real[:, 0] - fakes_mean)**2 ).sum() / ( (real[:, 0] - real[:, 0].mean())**2 ).sum() )
    rae  = np.abs( real[:, 0] - fakes_mean ).sum() / np.abs( real[:, 0] - real[:, 0].mean() ).sum()
    mape = 100. / len(real[:, 0]) * np.abs( 1. - fakes_mean/real[:, 0] ).sum()

    nlpd = nlpd_metric(fakes, real)
    nrmsep = nrmse_p_metric(fakes, real)
    picp_68 = picp_metric(fakes, real, alpha=0.68268)  # 1 sigma
    picp_95 = picp_metric(fakes, real, alpha=0.95450)  # 2 sigmas

    return [rmse, mae, rse, rae, mape, nlpd, nrmsep, picp_68, picp_95]


def get_quality_metrics_report(fakes, real):

    metrics_report = pd.DataFrame(columns=['RMSE', 'MAE', 'RSE', 'RAE', 'MAPE', 'NLPD', 'nRMSEp', 'PICP_68', 'PICP_95'])
    
    m = get_quality_metrics(fakes, real)
    metrics_report.loc[0, :] = m

    return metrics_report



def quality_metrics_report(df_fakes, df_real, segments):

    metrics_report = pd.DataFrame(columns=['RMSE', 'MAE', 'RSE', 'RAE', 'MAPE', 'NLPD', 'nRMSEp', 'PICP_68', 'PICP_95'])

    for aseg in segments:
        fakes = []
        for adf in df_fakes:
            ats = adf[adf['segment'] == aseg]
            fakes.append(ats['target'].values)
        fakes = np.array(fakes).T

        real = df_real[df_real['segment'] == aseg][['target']].values

        m = get_quality_metrics(fakes, real)
        metrics_report.loc[aseg, :] = m

    return metrics_report


def transform_format(data, segments):
    data_new = []
    for aseg in segments:
        df_aseg = data[['date', aseg]].dropna()
        adf = pd.DataFrame()
        adf['timestamp'] = pd.to_datetime(df_aseg['date'])
        adf['segment'] = [aseg] * len(df_aseg)
        adf['target'] = df_aseg[aseg]
        data_new.append(adf)
    data_new = pd.concat(data_new, axis=0)
    return data_new


def plot_predictions(df_known, df_pred, df_unknown, segments, saving_path = None):
    plt.figure(figsize=(15, 4*(len(segments)//2+1)))
    for i, aseg in enumerate(segments):
        plt.subplot(len(segments)//2+1, 2, i+1)
        ts_pred = df_pred[df_pred['segment'] == aseg]
        plt.plot(ts_pred['timestamp'], ts_pred['target'], label='imputed', color='C1')
        ts_known = df_known[df_known['segment'] == aseg]
        plt.plot(ts_known['timestamp'], ts_known['target'], label='known', color='C2')
        if df_unknown is not None:
            df_imputed = df_unknown.merge(df_pred, 'left', on=['timestamp', 'segment'])
            ts_imputed = df_imputed[df_imputed['segment'] == aseg]
            plt.scatter(ts_imputed['timestamp'], ts_imputed['target'], label='imputed', s=20, color='C3')
        plt.title(aseg)
        plt.legend()
    if saving_path is not None:
        plt.savefig(saving_path)
    plt.show()



def plot_intervals(df_known, df_imputed, segments, saving_path = None):
    plt.figure(figsize=(15, 4 * (len(segments) // 2 + 1)))
    for i, aseg in enumerate(segments):
        plt.subplot(len(segments) // 2 + 1, 2, i + 1)

        targets_imp = []
        for k, adf in enumerate(df_imputed):
            ts_imputed = adf[adf['segment'] == aseg]
            targets_imp.append(ts_imputed['target'].values)

        mean_imputed = np.mean(targets_imp, axis=0)
        std_imputed = np.std(targets_imp, axis=0)

        plt.plot(ts_imputed['timestamp'], mean_imputed, label=r'imputed ($\mu$)', color='C1')
        plt.fill_between(ts_imputed['timestamp'],
                         mean_imputed - 2 * std_imputed, mean_imputed + 2 * std_imputed,
                         label=r'imputed ($\mu \pm 2*\sigma$)', color='C1', alpha=0.2)

        ts_known = df_known[df_known['segment'] == aseg]
        plt.plot(ts_known['timestamp'], ts_known['target'], label='known', color='C2')

        plt.title(aseg)
        plt.legend()
    if saving_path is not None:
        plt.savefig(saving_path)
    plt.show()

    
def to_df_known(target):
    dfs = []
    for col in target.columns:
        df = target[[col]]
        df = df.reset_index()
        df['segment'] = col
        df = df.rename(columns={col: 'target', 'index': 'timestamp'})
        dfs.append(df)
    return pd.concat(dfs)


def to_target(df_known):
    dfs = []
    for col in df_known.segment.unique():
        df = df_known[df_known.segment == col][['timestamp', 'target']].copy()
        df = df.set_index('timestamp')
        df = df.rename(columns={'target': col})
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def robustness_checks(df_fakes, df_real, segments):
    reports = []
    for adf in df_fakes:
        arep = quality_metrics_report([adf], df_real, segments)
        arep = arep[['RMSE', 'MAE', 'RSE', 'RAE', 'MAPE']].values.astype(float)
        reports.append(arep)
    reports = np.array(reports)
    report_mean = pd.DataFrame(columns=['RMSE', 'MAE', 'RSE', 'RAE', 'MAPE'], 
                               data=reports.mean(axis=0), 
                               index=['log eurrub', 'log eurusd', 'log usdrub'])
    report_std = pd.DataFrame(columns=['RMSE', 'MAE', 'RSE', 'RAE', 'MAPE'], 
                               data=reports.std(axis=0), 
                               index=['log eurrub', 'log eurusd', 'log usdrub'])
    return report_mean, report_std