from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from utils import transform_format


class SegmentStandardScaler(object):

    def __init__(self, segments):
        """
        Standard Scaler for a table with known values. 
        
        Args:
        :param segments: array-like
            List of segment names in the table.
        """
        self.segments = segments
        self.scalers = {}

    def fit(self, df):
        """
        Args:
        :param df: pandas.DataFrame
            Table with known values.
            Required columns: 'timestamp', 'target', 'segment'.
        :return:
            self
        """
        
        for aseg in self.segments:
            df_seg = df[df['segment'] == aseg]
            scaler_seg = StandardScaler()
            scaler_seg.fit(df_seg[['target']].values)
            self.scalers[aseg] = scaler_seg

        return self

    def transform(self, df):
        """
        Args:
        :param df: pandas.DataFrame
            Table with known values.
            Required columns: 'timestamp', 'target', 'segment'.
        :return:
        df: pandas.DataFrame
            Scaled table with known values.
            Required columns: 'timestamp', 'target', 'segment'.
        """

        df_scaled = []
        for aseg in self.segments:
            df_seg = df[df['segment'] == aseg]
            scaler_seg = self.scalers[aseg]
            target_seg = scaler_seg.transform(df_seg[['target']].values)
            df_seg = df_seg[['timestamp', 'segment']]
            df_seg['target'] = target_seg
            df_scaled.append(df_seg)
        df_scaled = pd.concat(df_scaled, axis=0)
        df_scaled = pd.merge(df[['timestamp', 'segment']], df_scaled, 'left', on=['timestamp', 'segment'])

        return df_scaled

    def fit_transform(self, df):
        """
        Args:
        :param df: pandas.DataFrame
            Table with known values.
            Required columns: 'timestamp', 'target', 'segment'.
        :return:
        df: pandas.DataFrame
            Scaled table with known values.
            Required columns: 'timestamp', 'target', 'segment'.
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df):
        """
        Args:
        :param df: pandas.DataFrame
            Table with known values.
            Required columns: 'timestamp', 'target', 'segment'.
        :return:
        df: pandas.DataFrame
            Scaled table with known values.
            Required columns: 'timestamp', 'target', 'segment'.
        """

        df_scaled = []
        for aseg in self.segments:
            df_seg = df[df['segment'] == aseg]
            scaler_seg = self.scalers[aseg]
            target_seg = scaler_seg.inverse_transform(df_seg[['target']].values)
            df_seg = df_seg[['timestamp', 'segment']]
            df_seg['target'] = target_seg
            df_scaled.append(df_seg)
        df_scaled = pd.concat(df_scaled, axis=0)
        df_scaled = pd.merge(df[['timestamp', 'segment']], df_scaled, 'left', on=['timestamp', 'segment'])

        return df_scaled

    
def get_data_MCAR(df_name, df_len = 500, create_misses = False, share_of_misses = None):

    allowed_df = {'Electricity' : 'datasets\ETTM.csv',
                  'Solar' : 'datasets\solar.csv',
                  'Exchange': 'datasets\exchange.csv',
                  'AirQuality' : 'datasets\AirQuality.csv',
                  'ETTM' : 'datasets\ETTM.csv'}

    assert df_name in allowed_df, f'this df is not allowed, select one from {list(allowed_df.keys())}.'

    df = pd.read_csv(allowed_df[df_name])
    df['date'] =  np.arange(0,len(df),1)

    try:
        df = transform_format(df, segments = df.columns[:-1]).sort_values(['timestamp', 'segment']).reset_index(drop = True)
    except:
        df = transform_format(df, segments = df.columns[1:]).sort_values(['timestamp', 'segment']).reset_index(drop = True)
    df['target'] = df['target'].astype(str).str.replace(',','.',regex=True).astype(float)
    df['timestamp'] = df['timestamp'].values.astype(int)

    if create_misses:
        missing_values = np.random.choice(df.index.values, round(share_of_misses * len(df)))
        df['target'] = np.where(df.index.isin(missing_values), np.nan, df['target'] )

    return df[df.timestamp < df_len].sort_values(['timestamp', 'segment']).reset_index(drop = True)



def get_data_missing_intervals(df_name, df_len = 500, create_misses = False, length_of_intervals = None, num_of_intervals = None):
    
    allowed_df = {'Electricity' : 'datasets\Electricity.csv',
                  'Solar' : 'datasets\solar.csv',
                  'Exchange': 'datasets\exchange.csv',
                  'AirQuality' : 'datasets\AirQuality.csv',
                  'ETTM' : 'datasets\ETTM.csv'}
    
    assert df_name in allowed_df, f'this df is not allowed, select one from {list(allowed_df.keys())}.'
    
    df = pd.read_csv(allowed_df[df_name])
    df['date'] =  np.arange(0,len(df),1)
    try:
        df = transform_format(df, segments = df.columns[:-1]).sort_values(['timestamp', 'segment']).reset_index(drop = True)
    except:
        df = transform_format(df, segments = df.columns[1:]).sort_values(['timestamp', 'segment']).reset_index(drop = True)
        
    df['target'] = df['target'].astype(str).str.replace(';','',regex=True)
    df['target'] = df['target'].astype(str).str.replace(',','.',regex=True).astype(float)
    df['timestamp'] = df['timestamp'].values.astype(int)
    
    df = df[df.timestamp < df_len]

    if create_misses:
        for s in df.segment.unique():
            missing_values = np.random.choice(np.arange(0,np.max(df['timestamp']),1), num_of_intervals)
            for m in missing_values:
                df['target'] = np.where(((df.segment == s) & 
                                         (df.timestamp > m) & 
                                         (df.timestamp <= m+length_of_intervals)), np.nan, df['target'])

    return df.sort_values(['timestamp', 'segment']).reset_index(drop = True)