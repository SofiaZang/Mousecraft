import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

def plot_all_tseries(all_tseries, title=None, save_path=None):
    
    fig, axs = plt.subplots(len(all_tseries), 1, figsize=(5, len(all_tseries)), dpi=150)
    for i, tseries in enumerate(all_tseries):
        axs[i].plot(tseries)
    
        # remove all x and y axes and only keep the last one
        if i != len(all_tseries)-1:
            # remove x spline
            axs[i].spines['bottom'].set_visible(False)
            axs[i].set_xticks([])
        else:
            axs[i].set_xlabel('time')
            
        # remove left right and top spines as well as y ticks and labels
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].set_yticks([])

        axs[0].set_title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def plot_overlay_all_tseries(all_tseries_0, all_tseries_1, title=None, save_path=None):
        
    fig, axs = plt.subplots(len(all_tseries_0), 1, figsize=(5, len(all_tseries_0)), dpi=150)
    for i, tseries_0 in enumerate(all_tseries_0):
        tseries_1 = all_tseries_1[i]

        axs[i].plot(tseries_0, alpha=0.5, color='grey')
        axs[i].plot(tseries_1, alpha=0.6)
    
        # remove all x and y axes and only keep the last one
        if i != len(all_tseries_0)-1:
            # remove x spline
            axs[i].spines['bottom'].set_visible(False)
            axs[i].set_xticks([])
        else:
            axs[i].set_xlabel('time')
            
        # remove left right and top spines as well as y ticks and labels
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].set_yticks([])

        axs[0].set_title(title)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_lambda_optimisation(all_weight_decay, all_min_train_loss, all_val_loss, best_weight_decay, best_idx, all_train_pred, all_val_pred, y_train, y_val, all_train_loss_all_t, plot_fname_path=None):

    # make a mosaic plot
    mosaic = '''
        ACD
        BCD
        '''
    # make the C narrower than A and B with relative scaling
    fig, axs = plt.subplot_mosaic(mosaic, figsize=(13, 3), gridspec_kw={'width_ratios': [1, 0.5, 0.5]}) 
    axs['A'].plot(y_train, c='grey', label='y_true')
    axs['A'].plot(all_train_pred[best_idx], c='C2', alpha=0.7, label='y_true')
    axs['A'].set_ylim(-2.5, 2.5)
    # axs['A'].set_ylim(-1, 10)

    axs['B'].plot(y_val, c='grey', label='y_true')
    axs['B'].plot(all_val_pred[best_idx], alpha=0.7, c='C1', label='y_true')
    axs['B'].set_ylim(-2.5, 2.5)
    # axs['B'].set_ylim(-1, 10)

    # set xlim to be the same as A
    axs['B'].set_xlim(axs['A'].get_xlim())
    axs['B'].set_xlabel('time (frames)')
    axs['B'].set_ylabel('motion energy')
    # remove x axis in A
    axs['A'].set_xticks([])
    axs['A'].set_xlabel('')
    axs['A'].spines['bottom'].set_visible(False)

    axs['A'].set_title(f'weight_decay (optimal): {round(best_weight_decay,3)}')

    # normalize the losses by length of y_train and y_val
    all_min_train_loss = np.array(all_min_train_loss) / len(y_train)
    all_val_loss = np.array(all_val_loss) / len(y_val)

    axs['C'].plot(all_weight_decay, all_min_train_loss, c='C2', label='y_train_pred')
    axs['C'].plot(all_weight_decay, all_val_loss, c='C1', label='y_val_pred')
    axs['C'].set_xscale('log')
    axs['C'].set_yscale('log')
    axs['C'].set_xlabel('weight decay')
    axs['C'].set_ylabel('loss')
    axs['C'].set_title('loss over weight decay')
    # plot vertical line at current weight decay
    vline = axs['C'].axvline(best_weight_decay, c='k', alpha=0.5, label='y_true')
    vline = axs['C'].axvline(best_weight_decay, c='k', label='optimal $\lambda$')

    axs['C'].legend(frameon=False, fontsize=8)

    for (i, train_loss_all_t) in enumerate(all_train_loss_all_t):
        # increment alpha to make it lighter by taking into account the number of weight decays
        alpha = 0.1 + 0.9 * (i / len(all_train_loss_all_t))
        axs['D'].plot(np.array(train_loss_all_t).T, c='C3', alpha=alpha)

    axs['D'].set_xlabel('epoch')
    axs['D'].set_title('training loss over time')
    axs['D'].set_yscale('log')

    # remove top and right splines
    axs['A'].spines['top'].set_visible(False)
    axs['A'].spines['right'].set_visible(False)
    axs['B'].spines['top'].set_visible(False)
    axs['B'].spines['right'].set_visible(False)
    axs['C'].spines['top'].set_visible(False)
    axs['C'].spines['right'].set_visible(False)
    axs['D'].spines['top'].set_visible(False)
    axs['D'].spines['right'].set_visible(False)



    if plot_fname_path is not None:

        title = plot_fname_path.split('/')[-1].split('.')[0] # set title to the string between the last / and .png

        plt.suptitle(title)
        # move the suptitle a bit up
        plt.subplots_adjust(top=0.85)

        plt.savefig(plot_fname_path)

    plt.show()



# plot diagonal of matrix
def plot_mat_diag(cross_corr_mat_reg_pca1, marks = None, title='', cmap='Reds', save_path=None, ylim=[0, 1]):
    data = np.copy(np.diag(cross_corr_mat_reg_pca1))
    data[data < 0] = 0
    fig, axs = plt.subplots(1, 1, figsize=(1.6, 1.6), dpi=300)
    axs.plot(data, color='grey', alpha=0.2)
    # make sure the above plot is in the back
    axs.scatter(np.arange(len(data)), data, s=60, c=data, cmap=cmap, vmin=0, vmax=1)
  

    if marks is not None:
        # add marks above the first and last point of the scatter
        axs.plot(0, data[0]+0.15, marks[0], color='grey', markersize=5)
        axs.plot(len(data)-1, data[-1]+0.15, marks[1], color='grey', markersize=5)

    axs.set_xticks([0, cross_corr_mat_reg_pca1.shape[0]-1])
    axs.set_xticklabels(['P8', 'P13'])
    axs.set_yticks(ylim)
    axs.set_yticklabels(ylim)
    axs.set_xlabel('Session')
    axs.set_ylabel(title)
    axs.set_xlim(-0.7, cross_corr_mat_reg_pca1.shape[0]-0.3)

    # remove top and right spines
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)



def plot_mot_tseries_pair(mot1, mot2, tseries1, tseries2, label1=None, label2=None, n_avg=10, color='p', ylabel1='P8 → P9', ylabel2='P12 → P13', marks=None, title=None, save_path=None):

    if color == 'p':
        color = (238/255, 130/255, 238/255)
    elif color == 'g':
        color = (144/255, 238/255, 144/255)

    fig, axs = plt.subplots(2, 1, figsize=(5, 2), dpi=300)
    axs[0].plot(mot1, color='gray', alpha=0.5, label=label1)
    axs[0].plot(tseries1, color=color, alpha=0.7, label=label2)
    axs[0].set_ylabel(ylabel1, rotation=0)
    # remove all splines except yaxies
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)
    axs[0].spines['bottom'].set_visible(False)
    axs[0].spines['left'].set_visible(False)
    axs[0].set_yticks([])
    axs[0].set_yticklabels([])
    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    
    axs[1].plot(mot2, color='gray', alpha=0.5)
    axs[1].plot(tseries2, color=color, alpha=0.7)
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].set_yticks([])
    axs[1].set_yticklabels([])
    axs[1].set_xlabel('Time (min)')
    axs[1].set_xticks([0, len(mot1)/2, len(mot1)])
    axs[1].set_xticklabels([0, 10, 20])
    
    axs[1].set_ylabel(ylabel2, rotation=0)
    # add a title
    if label1 is not None and label2 is not None:
        # add a legend on top of the plot with the two labels side by side and no bounding box
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=2, frameon=False)
        
    if marks is not None:
        # add a triangle at y=0 and x = -0.1 at the beginning of the plot for first
        # and a circle for the second
        axs[0].plot(-30, 0.05, marks[0], color='grey', markersize=5)
        axs[1].plot(-30, 0.05, marks[1], color='grey', markersize=5)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_tseries_sessions(all_tseries, sessions, fs=30, title='', xlabel='', ylabel='', ylim=None, xlim=None, yline=None, save_path=None, show_r2=False):
    # all tseries can be a list of length sessions, each sublist will be ploted as overlayed time series
    # first if elements of alltseries a
    # are not lists then embed in another list (for later compatibility)
    if show_r2:
        all_r2 = [r2_score(t[1], t[0]) for t in all_tseries]

    all_tseries = all_tseries.copy() # make a copy to avoid changing the original list
    
    for i, tseries in enumerate(all_tseries):
        if not isinstance(tseries, list):
            all_tseries[i] = [tseries]
            
    
    fig, ax = plt.subplots(len(all_tseries), 1, figsize=(10/2, len(all_tseries)/2), dpi=300)
    
    for i, tseries in enumerate(all_tseries):
        for (j, ts) in enumerate(tseries):
            color = '#8CB49C' if j != 1 else 'grey'
            alpha = 1 if j != 1 else 0.5
            ax[i].plot(ts, color=color, alpha=alpha, linewidth=1)
        ax[i].set_ylim(ylim) if ylim is not None else None
        ax[i].set_xlim(xlim) if xlim is not None else None
        ax[i].set_yticks([])
        ax[i].set_yticklabels([])
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_ylabel(sessions[i])

        if show_r2:
            ax[i].text(0.95, 0.95, f'r2: {round(all_r2[i], 2)}', ha='right', va='top', transform=ax[i].transAxes, fontsize=6)
    
        if i == len(all_tseries)-1:
            ax[i].set_xlabel('Time (min)')
            ax[i].set_xticks([0, len(tseries[0])/2, len(tseries[0])-1])
            ax[i].set_xticklabels([0, int(len(tseries[0])/fs/(2*60)), int(len(tseries[0])/fs/60)])
        else:
            ax[i].set_xticks([])
            ax[i].set_xticklabels([])
            ax[i].spines['bottom'].set_visible(False)

    plt.suptitle(title)
    # save
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()
        

def plot_lambda_folds_days(all_mouse_w_decay_array, all_weight_decay, save_path=None):

    fig, axs = plt.subplots(1, len(all_mouse_w_decay_array), figsize=(2*len(all_mouse_w_decay_array), 2), sharey=True, sharex=True)
    for i, all_ds_w_decay in enumerate(all_mouse_w_decay_array):
        for j in range(all_ds_w_decay.shape[0]):
            axs[i].scatter([j]*all_ds_w_decay.shape[1]+ np.random.randn(all_ds_w_decay.shape[1])*0.05, all_ds_w_decay[j], c=f'C{j}', alpha=0.5)
            # add median in black
            axs[i].scatter(j, np.median(all_ds_w_decay[j]), c='k', marker='x')

        axs[i].set_yscale('log')
        axs[i].set_ylim(all_weight_decay[0], all_weight_decay[-1])
        axs[i].set_ylim(all_weight_decay[0]/1.5, all_weight_decay[-1]*1.5)
        
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)  

        axs[i].set_xticks(range(all_ds_w_decay.shape[0]))
        
        if i != 0:
            axs[i].spines['left'].set_visible(False)
            axs[i].yaxis.set_visible(False)
        else:
            axs[i].set_ylabel('lambda')
            axs[i].set_xlabel('outer fold idx')
        
        axs[i].set_title(f'ds {i}')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_cv_r2(all_mouse_r2_array, save_path=None):
    fig, axs = plt.subplots(1,2, figsize=(4, 2), dpi=300)

    for i in range(all_mouse_r2_array.shape[0]):
        axs[0].scatter(i*np.ones(all_mouse_r2_array.shape[1]), all_mouse_r2_array[i], c='C0', alpha=0.5)
        axs[0].scatter(i, np.median(all_mouse_r2_array[i]), c='k', marker='x')
        # same as above but restrict y axis between -0.1 and 1.1
        axs[1].scatter(i*np.ones(all_mouse_r2_array.shape[1]), all_mouse_r2_array[i], c='C0', alpha=0.5)
        axs[1].scatter(i, np.median(all_mouse_r2_array[i]), c='k', marker='x')
        axs[1].set_ylim(-0.1, 1.1)
        
    # remove all top and right spines
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # set y axis ticks and labels to 1 and 0 for second
    axs[1].set_yticks([0, 1])
    axs[1].set_yticklabels([0, 1])

    # set x axis ticks and lables to integers
    axs[0].set_xticks(range(all_mouse_r2_array.shape[0]))
    axs[0].set_xticklabels(range(all_mouse_r2_array.shape[0]))
    axs[1].set_xticks(range(all_mouse_r2_array.shape[0]))
    axs[1].set_xticklabels(range(all_mouse_r2_array.shape[0]))

    # add labels
    axs[0].set_ylabel('r2')
    axs[0].set_xlabel('dataset idx')

    plt.suptitle('r2 for each outer fold')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()