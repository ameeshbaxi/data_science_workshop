import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.calibration import calibration_curve
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import os

x_columns = ['hour', 'C1', 'banner_pos', 'site_id', 'site_domain',
             'site_category', 'app_id', 'app_domain', 'app_category',
             'device_id', 'device_ip', 'device_model', 'device_type',
             'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18',
             'C19', 'C20', 'C21']

# define the data types
dtypes = {'click' : np.int32, 'C1' : np.int64,
          'banner_pos' : np.int32, 'site_id' : np.int64 , 'site_domain' : np.int64,
          'site_category' : np.int64, 'app_id' : np.int64, 'app_domain' : np.int64,
          'app_category' : np.int64, 'device_id' : np.int64, 'device_ip' : np.int64,
          'device_model' : np.int64, 'device_type' : np.int64, 'device_conn_type' : np.int64,
          'C14' : np.int64, 'C15' : np.int64, 'C16' : np.int64, 'C17' : np.int64,
          'C18' : np.int64, 'C19' : np.int64, 'C20' : np.int64, 'C21' : np.int64}

# number of training sampling bins.
# This was worked out when train.csv file was processed in small chunks.
num_bins = 809
# The chunk size / bin size (for each files)
bin_size = 50000
chunk_size = bin_size
# number of test bins
num_test_bins = 92


def process_data_type(df):
    df.hour = df.hour.astype(str)
    df.hour = df.hour.str[:2] + '-' + df.hour.str[2:]
    df.hour = df.hour.str[:5] + '-' + df.hour.str[5:]
    df.hour = df.hour.str[:8] + ' ' + df.hour.str[8:]
    df.hour = pd.to_datetime(df.hour, format='%y-%m-%d %H')
    df.hour = df['hour'].apply(lambda x: x.toordinal())
    
    columns = ['site_id', 'site_domain', 'site_category',
               'app_id', 'app_domain', 'app_category',
               'device_id', 'device_ip', 'device_model']

    for c in columns:
        df[c] = df[c].apply(lambda x: int(x, 16))
    
    return df


def divide_csv_into_chuncks(csv_data):
    start = dt.datetime.now()
    chunksize = chunk_size
    j = 0
    index_start = 1
    dir_path = os.path.dirname(os.path.abspath(csv_data))
    csv_name = os.path.basename(csv_data).split('.')[0]
    for df in pd.read_csv(csv_data, chunksize=chunksize, iterator=True,
                          encoding='utf-8', low_memory=False):
        df = process_data_type(df)
        df.index += index_start
        j+=1
        file_name = os.path.join(dir_path, csv_name + '_chunk%d.csv' % j)
        df.to_csv(file_name, index=False)
        index_start = df.index[-1] + 1
        
        print('{} seconds: completed {} rows'.format((dt.datetime.now() - start).seconds, j*chunksize))
    
    return j


# helper function to read csv file
def read_data_csv(csv_file_name):
    df_sample = pd.read_csv(csv_file_name, low_memory=False, dtype=dtypes)
    # ensure that hour column is in ordinal format
    df_sample.hour = pd.to_datetime(df_sample.hour)
    df_sample.hour = df_sample['hour'].apply(lambda x: x.toordinal())
    return df_sample

# helper function to read csv file
def read_big_csv(csv_file_name):
    tp = pd.read_csv(csv_file_name, iterator=True, chunksize=100000,
                     dtype=dtypes, low_memory=False)
    df = pd.concat(tp, ignore_index=True)
    # ensure that hour column is in ordinal format
    df.hour = pd.to_datetime(df.hour)
    df.hour = df['hour'].apply(lambda x: x.toordinal())
    return df

# helper function to sample the training data
def sample_data(num_samples_per_bin=100, frac_pos=0.5):
    
    percent = frac_pos * 100
    percent_csv_file = '../data/train_sample_percent%d.csv' % percent
    
    if not os.path.exists(percent_csv_file):
        samples = []
        for i in range (1, num_bins + 1):
            df = pd.read_csv('../data/train_chunk%d.csv' % i, low_memory=False, dtype=dtypes)
            df.hour = pd.to_datetime(df.hour) 
            num_pos = int(frac_pos * num_samples_per_bin)
            num_neg = int((1 - frac_pos) * num_samples_per_bin)
            if len(df.index) != bin_size:
                bin_frac = 1.0 * len(df.index) / bin_size
                # at least get one sample :P
                num_pos = int(num_pos * bin_frac + 1)
                num_neg = int(num_neg * bin_frac + 1)
            
            samples.append(df[df.click > 0].sample(num_pos))
            samples.append(df[df.click == 0].sample(num_neg))
    
        df_sample = pd.concat(samples, ignore_index=True)
        df_sample.to_csv(percent_csv_file, index=False)

    
    return read_data_csv(percent_csv_file)

# helper function to sample the test data
def sample_test_data(num_samples_per_bin=900):
    
    test_sample_csv = '../data/test_sample.csv'
    
    if not os.path.exists(test_sample_csv):
        
        samples = []
        for i in range (1, num_test_bins + 1):
            df = pd.read_csv('../data/test_chunk%d.csv' % i, low_memory=False, dtype=dtypes)
            df.hour = pd.to_datetime(df.hour) 
            num_samples = int(num_samples_per_bin)
            if len(df.index) != bin_size:
                bin_frac = 1.0 * len(df.index) / bin_size
                # at least get one sample :P
                num_samples = int(num_samples_per_bin * bin_frac + 1)
                
            samples.append(df.sample(num_samples))
            
        test_sample = pd.concat(samples, ignore_index=True)
        test_sample.to_csv(test_sample_csv, index=False)

    
    return read_data_csv(test_sample_csv)

def train_data(frac_pos=0.5):
    
    percent = frac_pos * 100
    percent_csv_file = r'../data/train_percent%d.csv' % percent
    
    if not os.path.exists(percent_csv_file):
        tp = pd.read_csv(r'../data/train', iterator=True, chunksize=100000,
                         low_memory=False, encoding='utf-8')
        df = pd.concat(tp, ignore_index=True)
        df = process_data_type(df)

        samples = []
        num_pos = df[df.click > 0].count()[0]
        num_neg = ((1 - frac_pos) / frac_pos) * 1.0 * num_pos
        samples.append(df[df.click > 0].sample(num_pos))
        samples.append(df[df.click == 0].sample(num_neg))
        df_sample = pd.concat(samples, ignore_index=True)
        df_sample.reindex(np.random.permutation(df_sample.index))
        print('Writing to csv file (%s)' % percent_csv_file)
        df_sample.to_csv(percent_csv_file, index=False)

    return read_big_csv(percent_csv_file)

def test_data():
    processed_csv_file = '../data/test_processed.csv'
    
    if not os.path.exists(processed_csv_file):
        tp = pd.read_csv('../data/test', iterator=True, chunksize=100000,
                         low_memory=False, encoding='utf-8')
        df = pd.concat(tp, ignore_index=True)
        df = process_data_type(df)
        df.to_csv(processed_csv_file, index=False)

    return read_big_csv(processed_csv_file)
    
def calibration_plot(all_clfs, X, y):
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf_, name in all_clfs:
        if hasattr(clf_, "predict_proba"):
            prob_pos = clf_.predict_proba(X)[:, 1]
        else:  # use decision function
            prob_pos = clf_.decision_function(X)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name, ))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.show()
    
def plot_sample(sample_df):
    pca = PCA(n_components=2)
    sample_pca = sample_df.copy()
    transformed = pca.fit_transform(sample_pca[x_columns])
    sample_pca['x'] = transformed[:,0]
    sample_pca['y'] = transformed[:,1]

    sns.regplot(sample_pca[sample_pca.click > 0].x, y=sample_pca[sample_pca.click > 0].y, label='CLICK', fit_reg=False)
    sns.regplot(sample_pca[sample_pca.click == 0].x, y=sample_pca[sample_pca.click == 0].y, label='NO CLICK', fit_reg=False)
    plt.legend()
    plt.show()
    

def plot_distribution(Xdata, ydata, clf, predicted=True, title=None):
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(Xdata)
    df = pd.DataFrame(index=range(len(Xdata)))
    df['x'] = transformed[:,0]
    df['y'] = transformed[:,1]
    if predicted:
        df['click'] = clf.predict(Xdata)
    else:
        df['click'] = ydata
    
    sns.regplot(df[df.click > 0].x, y=df[df.click > 0].y, label='CLICK', fit_reg=False)
    sns.regplot(df[df.click == 0].x, y=df[df.click == 0].y, label='NO CLICK', fit_reg=False)
    if title is not None:
        sns.plt.title(title)
    plt.legend()
    plt.show()

