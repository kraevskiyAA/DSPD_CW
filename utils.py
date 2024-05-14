from sklearn.preprocessing import StandardScaler
import warnings
import numpy as np
warnings.filterwarnings("ignore")
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms

def normalize_timestamp(timestamp):
    """
    This functions reformat timestamp series to POSIX seconds and normalizes output vector
    :param timestamp: pd.Series with data type of datetime64
    :param sc: sklearn.preprocessing.StandardScaler.
    If None, fit new one, and transform the data

    :return: numpy array with scaled times in seconds
    """
    _tt = timestamp.astype('int64').values.reshape(-1, 1)
    if not sc:
        sc = StandardScaler()
        ret = sc.fit_transform(_tt)
    else:
        ret = sc.transform(_tt)
    return ret, sc

def parse_data(x, segments):
    # extract the last value for each attribute
    x = x.set_index("segment").to_dict()["target"]

    values = []

    for attr in segments:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    return values

def preproc_dataset(data, missing_ratio = 0.1):

    observed_values = []
    observed_times = []

    segments = data.segment.unique()

    for h in np.sort(data.timestamp.unique()):
        observed_values.append(parse_data(data[data["timestamp"] == h], segments))
        times = data[data['timestamp'] == h]['timestamp']
        observed_times.append(times.iloc[-1] if len(times) > 0 else np.nan)
    observed_values = np.array(observed_values)
    observed_times = np.array(observed_times)
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_times = np.nan_to_num(observed_times)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")

    return observed_values, observed_masks, gt_masks, observed_times



class Custom_Dataset(Dataset):
    def __init__(self, data, length_of_sequence = 20,  use_index_list=None, eval_length=48, missing_ratio=0.0, seed=0, use_rolling_window = True):

        self.eval_length = eval_length
        self.segment_num = data.segment.nunique()
        np.random.seed(seed)  # seed for ground truth choice

        times = np.sort(data['timestamp'].unique())

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []
        self.observed_times = []
        if use_rolling_window:
            for t in range(len(times) - length_of_sequence+1):
                observed_values, observed_masks, gt_masks, observed_times = preproc_dataset(
                   data[data.timestamp.isin(times[t:t+length_of_sequence])], missing_ratio = missing_ratio)

                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
                self.observed_times.append(observed_times)
                
        else:
            required_windows = int(np.ceil(len(times) / length_of_sequence))
            for t in range(required_windows):
                if t != required_windows - 1:
                    observed_values, observed_masks, gt_masks, observed_times = preproc_dataset(
                       data[data.timestamp.isin(times[t*(length_of_sequence):(t+1) *length_of_sequence])], missing_ratio = missing_ratio)
                else:
                    observed_values, observed_masks, gt_masks, observed_times = preproc_dataset(
                       data[data.timestamp.isin(times[-length_of_sequence:])], missing_ratio = missing_ratio)
                                        
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
                self.observed_times.append(observed_times)

        self.observed_values = np.array(np.asarray(self.observed_values, dtype= "float"))
        self.observed_masks = np.array(np.asarray(self.observed_masks, dtype= "float"))
        self.gt_masks = np.array(np.asarray(self.gt_masks, dtype= "float"))
        self.observed_times = np.array(np.asarray(self.observed_times))

        # calc mean and std and normalize values
        # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        tmp_values = self.observed_values.reshape(-1, self.segment_num)
        tmp_masks = self.observed_masks.reshape(-1, self.segment_num)
        mean = np.zeros(self.segment_num)
        std = np.zeros(self.segment_num)

        for k in range(self.segment_num):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()
        self.observed_values = (
            (self.observed_values - mean) / std * self.observed_masks
        )

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": from_numpy(self.observed_values[index].T),
            "observed_mask": from_numpy(self.observed_masks[index].T),
            "gt_mask": from_numpy(self.gt_masks[index].T),
            "timepoints": from_numpy(np.arange(self.eval_length)),
            "times": from_numpy(self.observed_times[index]),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)
    

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