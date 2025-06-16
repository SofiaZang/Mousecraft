import numpy as np
import matplotlib.pyplot as plt
import torch

def zscore(X, axis=1):
    """Z-score normalization across specified axis (default: across time)."""
    return (X - np.mean(X, axis=axis, keepdims=True)) / np.std(X, axis=axis, keepdims=True)

class MoveDeveDataset():
    def __init__(self, X, y):
        self.X = zscore(X, axis=1) # dF/F
        # replace nan values with zeros
        self.X[np.isnan(self.X)] = 0
        self.y = y
        self.n = X.shape[1]

        print(f'Created dataset with {self.n} samples')
    
    def get_subset(self, out_i, in_i, k=10, plot_subset=True):
        # get subset of data for cross-validation based on outer and inner fold indices

        self.outer_fold_size = int(self.n / k) # size of each fold (number of samples
        self.inner_fold_size = int((self.n-self.outer_fold_size)/k) # size of each fold (number of samples


        # get the test set indices from outer fold
        test_idx = np.arange(out_i*self.outer_fold_size, (out_i+1)*self.outer_fold_size)

        # get the validation and training set indices from inner fold
        # all indices except test indices
        trainval_idx = np.arange(self.n)
        trainval_idx = np.delete(trainval_idx, test_idx)

        # get the validation set indices from inner fold
        val_idx_in_trainval = np.arange(in_i*self.inner_fold_size, (in_i+1)*self.inner_fold_size)
        val_idx = trainval_idx[val_idx_in_trainval]

        train_idx = np.delete(trainval_idx, val_idx_in_trainval)
        # use the indices to get the data (also add plot to make sure it's correct)
        
        X_test = self.X[:, test_idx]
        y_test = self.y[test_idx]

        X_val = self.X[:, val_idx]
        y_val = self.y[val_idx]

        X_train = self.X[:, train_idx]
        y_train = self.y[train_idx]


        # make a plot as sanity check to make sure it worked
        if plot_subset:
            self.plot_subset_split(test_idx, val_idx, train_idx, y_test, y_val, y_train, title=f'Outer fold {out_i}, inner fold {in_i}')
        
        # convert to torch tensors
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_val = torch.tensor(X_val).float()
        y_val = torch.tensor(y_val).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()

        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    
    def plot_subset_split(self, test_idx, val_idx, train_idx, y_test, y_val, y_train, title=None):

        fig, ax = plt.subplots(2, 1, figsize=(5, 2.5), dpi=100, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax[0].imshow(self.X, aspect='auto', cmap='Greys', vmin=0, vmax=2.56)

        # shading of raster
        test_mask = np.zeros(self.X.shape)
        test_mask[:, test_idx] = 1
        val_mask = np.zeros(self.X.shape)
        val_mask[:, val_idx] = 1
        train_mask = np.zeros(self.X.shape)
        train_mask[:, train_idx] = 1

        ax[0].imshow(test_mask, aspect='auto', cmap='Blues', alpha=0.2, vmin=0, vmax=2)
        ax[0].imshow(val_mask, aspect='auto', cmap='Oranges', alpha=0.2, vmin=0, vmax=2)
        ax[0].imshow(train_mask, aspect='auto', cmap='Greens', alpha=0.2, vmin=0, vmax=2)

        ax[0].set_ylabel('cells')
        ax[0].set_title(title)
        # get time values for the y
        
        ax[1].plot(test_idx, y_test, label='test')
        ax[1].plot(val_idx, y_val, label='val')
        ax[1].plot(train_idx, y_train, label='train')

        ax[1].set_ylabel('motion energy')
        ax[1].set_xlabel('time (frames)')
        # make small legend to fit into small plot
        ax[1].legend(loc='upper right', fontsize=3)
        plt.show()

    def get_chunked_subset(self, out_i, in_i, k=4, n_chunks=5, plot_subset=True):
        
        t_chunk = int(self.n / n_chunks)
        t_test_chunk = int(t_chunk / k)
        t_trainval_chunk = t_chunk - t_test_chunk
        t_val_chunk = int(t_trainval_chunk / k)
        t_train_chunk = t_trainval_chunk - t_val_chunk


        # print each in individual row
        # print(f't_chunk: {t_chunk}\nt_test_chunk: {t_test_chunk}\nt_trainval_chunk: {t_trainval_chunk}\nt_val_chunk: {t_val_chunk}\nt_train_chunk: {t_train_chunk}')

        # for each chunk create the same subdivision as in get_subset and concatenate the indices for train val and test

        # initialise empty arays that will be concatenated to
        test_idx = np.array([])
        val_idx = np.array([])
        train_idx = np.array([])

        for i in range(n_chunks):
            # print(f'\nChunk {i}')
            idx_chunk = np.arange(i*t_chunk, (i+1)*t_chunk)
            # print(f'idx_chunk: {idx_chunk[0]} - {idx_chunk[-1]}')

            test_idx_chunk = np.arange(i*t_chunk + out_i*t_test_chunk, i*t_chunk + (out_i+1)*t_test_chunk)
            # print(f'test_idx_chunk: {test_idx_chunk[0]} - {test_idx_chunk[-1]}')

            # Calculate the indices of test_idx_chunk in idx_chunk
            test_idx_chunk_indices = np.where(np.isin(idx_chunk, test_idx_chunk))[0]

            # Delete those indices from idx_chunk
            trainval_idx_chunk = np.delete(idx_chunk, test_idx_chunk_indices)
            # print(f'trainval_chunk: {trainval_idx_chunk[0]} - {trainval_idx_chunk[-1]}')

            # index val_idx_chunk from trainval_idx_chunk
            val_idx_chunk = trainval_idx_chunk[in_i*t_val_chunk:(in_i+1)*t_val_chunk]
            # print(f'val_idx_chunk: {val_idx_chunk[0]} - {val_idx_chunk[-1]}')

            # Calculate the indices of val_idx_chunk in trainval_idx_chunk
            val_idx_chunk_indices = np.where(np.isin(trainval_idx_chunk, val_idx_chunk))[0]

            # Delete those indices from trainval_idx_chunk
            train_idx_chunk = np.delete(trainval_idx_chunk, val_idx_chunk_indices)
            # print(f'train_idx_chunk: {train_idx_chunk[0]} - {train_idx_chunk[-1]}')

            # concatenate the indices
            test_idx = np.concatenate((test_idx, test_idx_chunk)).astype(int)
            val_idx = np.concatenate((val_idx, val_idx_chunk)).astype(int)
            train_idx = np.concatenate((train_idx, train_idx_chunk)).astype(int)
            
        test_idx = test_idx[:, np.newaxis]
        
        X_test = self.X[:, test_idx]
        y_test = self.y[test_idx]

        X_val = self.X[:, val_idx]
        y_val = self.y[val_idx]

        X_train = self.X[:, train_idx]
        y_train = self.y[train_idx]


        # make a plot as sanity check to make sure it worked
        if plot_subset:
            self.plot_subset_split(test_idx, val_idx, train_idx, y_test, y_val, y_train, title=f'Outer fold {out_i}, inner fold {in_i}')

        # convert to torch tensors
        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_val = torch.tensor(X_val).float()
        y_val = torch.tensor(y_val).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()

        return X_train, y_train, X_val, y_val, X_test, y_test
    

def dechunk_cv_prediction(all_mouse_test_pred, all_mouse_test_truth, k=5, n_chunks=10):

    fold_length = len(all_mouse_test_pred[0]) // (k)
    chunk_length = len(all_mouse_test_pred[0]) // (n_chunks)
    test_chunk_length = len(all_mouse_test_pred[0]) // (n_chunks * k)
    
    all_mouse_test_pred_dechunked = []
    all_mouse_test_truth_dechunked = []

    for (i, test_pred) in enumerate(all_mouse_test_pred):
        test_truth = all_mouse_test_truth[i]

        test_pred_dechunked = np.zeros_like(test_pred)
        test_truth_dechunked = np.zeros_like(test_truth)

        for j in range(k): # for each outer fold
            
            # define indices where the test starts and ends for this fold
            fold_test_indeces = np.zeros(len(test_pred))
            
            # fold-dependednt shift index (each time it is shifted by test_chunk_length)
            fold_shift = j * test_chunk_length

            for l in range(n_chunks):
                fold_chunk_start_idx = l * chunk_length + fold_shift
                fold_test_indeces[fold_chunk_start_idx:fold_chunk_start_idx + test_chunk_length] = 1

                fold_test_indeces = fold_test_indeces.astype(bool)
        
            fold_chunk_start_idx = j * fold_length
            fold_chunk_end_idx = fold_chunk_start_idx + fold_length


            test_pred_chunk = test_pred[fold_chunk_start_idx:fold_chunk_end_idx]
            test_truth_chunk = test_truth[fold_chunk_start_idx:fold_chunk_end_idx]

            test_pred_dechunked[fold_test_indeces] = test_pred_chunk
            test_truth_dechunked[fold_test_indeces] = test_truth_chunk

    
        all_mouse_test_pred_dechunked.append(test_pred_dechunked)
        all_mouse_test_truth_dechunked.append(test_truth_dechunked)

    

    return all_mouse_test_truth_dechunked, all_mouse_test_pred_dechunked

def plot_dechunking(all_beh, all_mouse_test_truth, all_mouse_test_truth_dechunked):
    plt.figure(figsize=(10,3))
    plt.plot(all_beh[0], label='beh')
    plt.plot(all_mouse_test_truth[0]+10, label='chunked')
    plt.plot(all_mouse_test_truth_dechunked[0]+20, label='dechunked')
    plt.legend()
    plt.show()