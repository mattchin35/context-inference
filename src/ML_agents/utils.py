import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
import matplotlib.cm as cm
import pickle

def save_parameters(opts, save_name):
    cur_dict = opts.__dict__
    cur_dict = {k: v for k, v in cur_dict.items() if k[:2] != '__'}

    # save_name = save_name + '_' + opts.losses
    with open(save_name + '.json', 'w') as f:
        json.dump(cur_dict, f)

def load_parameters(save_path):
    """Load config."""
    with open(os.path.join(save_path + '.json'), 'r') as f:
        config_dict = json.load(f)

    class Config():
        pass

    config = Config()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config

def load_results(root_dir, subfile=None):
    dir = os.path.join(root_dir, 'files')
    dirs = [os.path.join(root_dir, dir, n) for n in os.listdir(dir)]
    dirs = sorted(dirs)
    mse_loss = []
    loss = []
    config = []
    for i, d in enumerate(dirs):
        if subfile is not None:
            d = os.path.join(d,subfile)
        config.append(load_parameters(os.path.join(d, 'parameters')))

        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        mse_loss.append(log['mse_loss'])
        loss.append(log['loss'])
    return xe_loss, loss, config

def subimage_easy(tup, col, row, save_name, cbar=False, vmin=-1, vmax=1, tight_axes=False, image=False, orderC=True,
                  ax_op=['off', 'image']):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    for t, w in tup:
        if row > 1 and col > 1:
            cur_axis = ax[r, c]
        elif row > 1 or col > 1:
            cur_axis = ax[c]
        else:
            cur_axis = ax

        if len(w.shape) < 2:
            w = np.expand_dims(w, 0)

        plt.sca(cur_axis)
        # sns.heatmap(w, cmap='RdBu_r', ax=ax[r, c], cbar=cbar, center=0)
        sns.heatmap(w, cmap='RdBu_r', vmin=vmin, vmax=vmax, ax=cur_axis, cbar=cbar, center=0)
        if t != '':
            cur_axis.set_title(t)

        # cur_axis.axis('off')
        # if image:
        #     cur_axis.axis('image')

        if ax_op:  # ['off', 'image'], etc
            for op in ax_op:
                cur_axis.axis(op)

        if orderC:
            c += 1
            if c >= col:
                r += 1
                c = 0
        else:
            r += 1
            if r >= row:
                c += 1
                r = 0

    if tight_axes:
        plt.tight_layout()
    fig.savefig(save_name, bbox_inches='tight', figsize=(14, 10), dpi=500)
    plt.close()

def subplot_easy(tup, col, row, save_name, xlim=None, ylim=None, hide_ticks=False, ax_op=[],
                 suptitle=None, legends=None, scatter=False, tight_axes=False, scatter_size=None, orderC=True):
    """input: list of tuples
    first arg of each tuple: title
    second arg of each tuple: matrix to plot
    """
    if legends:
        colors = cm.jet(np.linspace(0, 1, len(legends)))
        cyc = plt.cycler('color', colors)
        # cmap = plt.get_cmap('jet')
        # colors = [cmap(i) for i in np.linspace(0, 1, len(legends))]

    c, r = 0, 0
    fig, ax = plt.subplots(nrows=row, ncols=col)
    if suptitle:
        plt.suptitle(suptitle)

    def multiplot(ax, mat):
        try:
            if len(np.squeeze(mat).shape) == 1:
                ax.plot(mat, linewidth=.5)
            else:
                if scatter:
                    ax.scatter(mat[:, 0], mat[:, 1], s=scatter_size)
                else:
                    ax.plot(mat[:, 0], mat[:, 1], linewidth=.5)

        except TypeError:
            print(mat)
            print("Plotting error - pass the data to be plotted as a numpy array or as a list of numpy arrays.")

    for t, w in tup:
        if row > 1 and col > 1:
            cur_axis = ax[r, c]
        elif row > 1 or col > 1:
            cur_axis = ax[c]
        else:
            cur_axis = ax

        if isinstance(w, list) or isinstance(w, tuple):
            # multiple data for each plot
            for mat in w:
                multiplot(cur_axis, mat)
        else:
            multiplot(cur_axis, w)

        if legends:
            cur_axis.legend(legends)

        if t:  # t can be '', None for False
            cur_axis.set_title(t)
        if xlim:
            cur_axis.set_xlim(xlim[0], xlim[1])
        if ylim:
            cur_axis.set_ylim(ylim[0], ylim[1])
        if hide_ticks:
            cur_axis.set_xticklabels([])
            cur_axis.set_yticklabels([])
            cur_axis.set_xticks([])
            cur_axis.set_yticks([])
        if ax_op:
            for op in ax_op:
                cur_axis.axis(op)

        if orderC:
            c += 1
            if c >= col:
                r += 1
                c = 0
        else:
            r += 1
            if r >= row:
                c += 1
                r = 0

    if tight_axes:
        plt.tight_layout()
    fig.savefig(save_name + '.png', bbox_inches='tight', figsize=(10, 10), dpi=500)
    plt.close()

def adjust(ax):
    # plt.style.use('dark_background')
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    # ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # ax.yaxis.set_ticks([0, 0.5, 1.0])
    # ax.set_ylim([0, 1])

def plot_weights(data, save_path, ylabel ='', xlabel= ''):
    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    # plt.style.use('dark_background')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    adjust(ax)
    vlim = np.round(np.max(abs(data)), decimals=1)
    im = ax.imshow(data, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, data.shape[1]])
    ax.set_yticks([0, data.shape[0]])
    plt.axis('scaled')
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.savefig(os.path.join('./lab_meeting/images/' + save_path + '.png'),
                transparent=True)

def plot_weights_simple(data, save_path, ylabel='',xlabel='', vmin= None, vmax= None):
    print(plt.style.available)
    rect = [0.15, 0.15, 0.65, 0.65]
    # plt.style.use('seaborn-bright')
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = fig.add_axes(rect)

    adjust(ax)
    if vmin is None or vmax is None:
        vmax = np.round(np.max(abs(data)), decimals=1)
        vmin = -vmax

    im = ax.imshow(data, cmap= 'RdBu_r', vmin=vmin, vmax=vmax,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, data.shape[1]-1])
    ax.set_yticks([0, data.shape[0]-1])
    plt.tick_params(axis='both', which='major')
    # plt.axis('image')
    plt.savefig(save_path + '.png')


