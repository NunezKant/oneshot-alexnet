import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from scipy.io import loadmat
#from approxineuro import oneshot_utils
from scipy import io
import random

def load_nat30k(data_path, mouse_name, exp_date, ilayer=2, highvar=True, normalize=False):
    """
    Load nat30k dataset.
    ilayer: int, 1 or 2, layer 1 or layer 2/3.
    returns
    spks: np.array, shape (n_stim, n_neurons)
    normalize: bool, if True, zscore the data across time.
    """
    # load stim
    file_path = os.path.join(data_path, f"{mouse_name}_{exp_date}_nat30k.npz")
    dat = np.load(file_path, allow_pickle=True)
    spks = dat['S']
    istim = dat['istim']
    xpos, ypos = dat['xpos'], dat['ypos']
    csig = dat['csig']
    layer = dat['layer']
    idx = np.where(layer==ilayer)[0]
    spks = spks[idx]
    xpos, ypos = xpos[idx], ypos[idx]
    csig = csig[idx]
    # print(spks.shape, istim.shape, xpos.shape, ypos.shape, csig.shape, idx.shape)
    print('mean csig of all neurons', csig.mean())
    if highvar:
        idx = (csig>csig.mean())
        spks = spks[idx]
        xpos, ypos = xpos[idx], ypos[idx]
        csig = csig[idx]
        print('choosing high csig neurons')
        print(f"mean csig of {sum(idx)} chosen neurons: {csig.mean():.2f}")
    spks = spks.T
    if normalize:
        spks_mean = spks.mean(axis=0)
        spks -= spks_mean
        spks /= spks.std(axis=0)
        print('normalized')
    return spks, istim, xpos, ypos, csig


def load_neural_data_nat30k(neural_data_path, layer_id='23', highvar=True, normalize=True):
    """
    For nat30k recordings.
    Loads neural data from a file and zscore, return indexes of neurons with high signal variance is highvar is True.
    layer_id: str, layer id of the layer to load, either '23' or '1'.
    """
    # load neurons
    # os.path.join(neural_root, f'{session}_nat30k.npz')
    dat = np.load(neural_data_path, allow_pickle=True) # neural data
    iplane = dat['iplane']

    # neurons_atframes, xpos, ypos, iplane, subset_stim, csig = load_neurons_raw(mname, expdate, blk, datapath)

    if layer_id == '1':
        ilayer = np.where(iplane>=10)[0]
    else:
        ilayer = np.where(iplane<10)[0]
    print(f'{len(ilayer)} layer{layer_id} neurons')

    istim = dat['istim'].astype('int') - 1 - 32 # here only for the file including both 8x4 and 30k dataset
    ypos, xpos = dat['ypos'][ilayer], dat['xpos'][ilayer]
    csig = dat['csig'][ilayer]
    spks = dat['S'][ilayer]

    from scipy.stats import zscore
    spks = spks.T
    if normalize:
        spks_mean = spks.mean(axis=0)
        spks -= spks_mean
        spks /= spks.std(axis=0)
    spks_raw = spks.copy()
    if highvar:
        igood = csig > csig.mean()
        spks = spks[:, igood]
    else:
        igood = np.ones(len(csig), dtype=bool)
    sigvar = csig[igood]
    sigvar_all = csig.copy()
    print(f'mean signal variance of all neurons {len(igood)}: {sigvar_all.mean():.2f}')
    print(f'mean signal variance of {np.sum(igood)} chosen neurons: {sigvar.mean():.2f}')

    return xpos, ypos, spks_raw, istim, csig, igood

def load_neural_data_nat30k_new(mname, expdate, blk='1', layer_id='23', highvar=True, zscore=True, gcamp8=False):
    """
    load neural data from raw recording on the server.
    For nat30k recordings.
    Loads neural data from a file and zscore, return indexes of neurons with high signal variance is highvar is True.
    layer_id: str, layer id of the layer to load, either '23' or '1'.
    """
    # load neurons
    # os.path.join(neural_root, f'{session}_nat30k.npz')
    datapath = f"data/{mname}_{expdate}_{blk}_raw_data.npz"
    neurons_atframes, xpos, ypos, iplane, subset_stim, csig = load_neurons_raw(mname, expdate, blk, datapath, gcamp8=gcamp8)

    if layer_id == '1':
        ilayer = np.where(iplane>=int((iplane.max()+1)/2))[0]
    else:
        ilayer = np.where(iplane<int((iplane.max()+1)/2))[0]
    xpos, ypos = xpos[ilayer], ypos[ilayer]
    print(f'{len(ilayer)} layer{layer_id} neurons')

    istim = subset_stim.astype('int') - 1 - 32 # here only for the file including both 8x4 and 30k dataset
    i_nat30k = np.where(istim>=0)[0]
    istim = istim[i_nat30k]
    neurons_atframes = neurons_atframes[:, i_nat30k]
    csig = csig[ilayer]
    spks = neurons_atframes[ilayer]
    if zscore:
        spks = spks.T
        spks_mean = spks.mean(axis=0)
        spks -= spks_mean
        spks /= spks.std(axis=0)
        spks = spks.T
    if highvar:
        igood = csig > csig.mean()
        spks = spks[igood]
    else:
        igood = np.ones(len(csig), dtype=bool)
    sigvar = csig[igood]
    sigvar_all = csig.copy()
    print(f'mean signal variance of all neurons {len(igood)}: {sigvar_all.mean():.2f}')
    print(f'mean signal variance of {np.sum(igood)} chosen neurons: {sigvar.mean():.2f}')

    return xpos, ypos, spks, istim, csig, igood

def split_data_with_repeats(istim, ntrain=0.8, nval=0.1, ntest=0.1, seed=0):
    # (IMPORTANT) put all repeated data into training set
    # test and val should be unseen stimuli
    # random.seed(seed)
    # np.random.seed(seed)
    # ist, reps = np.unique(istim, return_counts=True)
    # ireps = np.isin(istim, ist[reps > 1])
    # n_reps = ireps.sum()
    # arg_nonreps = np.where(~ireps)[0]
    # n_stim = istim.shape[0]

    # itrain = np.zeros(n_stim, 'bool')
    # arg_itrain = np.random.choice(arg_nonreps, int(ntrain*n_stim - n_reps), replace=False)
    # itrain[arg_itrain] = True
    # itrain[ireps] = True

    # arg_test = np.where(~itrain)[0]
    # arg_test = np.random.choice(arg_test, int(ntest*n_stim), replace=False)
    # itest = np.zeros(n_stim, 'bool')
    # itest[arg_test] = True
    # ival = ~itrain & ~itest

    ist, reps = np.unique(istim, return_counts=True)
    ireps = np.isin(istim, ist[reps > 1])
    n_reps = ireps.sum()
    n_stim = istim.shape[0]

    train_frac = ntrain - (n_reps / n_stim)
    random.seed(seed)
    np.random.seed(seed)
    itrain = np.random.rand(n_stim - n_reps) < train_frac
    itest = ~itrain
    itest[itest>0] = np.random.rand(itest.sum()) > (nval/(nval+ntest))
    ival = np.logical_and(~itrain, ~itest)

    inds = [itrain, ival, itest]
    inds0 = []
    for ind in inds:
        ind0 = np.zeros(n_stim, 'bool')
        ind0[~ireps] = ind
        inds0.append(ind0)
    itrain, ival, itest = inds0
    itrain[ireps] = True
    return itrain, ival, itest

def load_neurons_raw(mname, expdate, blk, file_path, iexp=0, gcamp8=False):
    """ load raw neural data.
    mname: mouse name
    expdate: experiment date
    blk: block number
    file_path: path to data file
    """
    # file_path = f"../data/{mname}_{expdate}_{blk}_raw_data.npz"
    if os.path.exists(file_path):
        dat = np.load(file_path, allow_pickle=True)
        spks = dat['spks']
        xpos = dat['xpos']
        ypos = dat['ypos']
        iplane = dat['iplane']
        stat = dat['stat']
        ops = dat['ops']
        timeline = dat['timeline']
        print("Loaded data from file")
    else:
        exp_db = [{"mname": mname, "datexp": expdate, "blk": blk}]
        root = os.path.join("/home/stringlab/dm11_pachitariu/data/PROC", mname, expdate, blk)

        fname = "Timeline_%s_%s_%s" % (mname, expdate, blk)
        fnamepath = os.path.join(root, fname)

        timeline = io.loadmat(fnamepath, squeeze_me=True)["Timeline"]

        spks, xpos, ypos, iplane, stat, ops = oneshot_utils.load_neurons(exp_db[iexp], dual_plane=True, baseline=True)

        dat = {}
        dat['spks'] = spks
        dat['xpos'] = xpos
        dat['ypos'] = ypos
        dat['iplane'] = iplane
        dat['stat'] = stat
        dat['ops'] = ops
        dat['timeline'] = timeline
        np.savez(f"data/{mname}_{expdate}_{blk}_raw_data.npz", **dat)
    if gcamp8:
        neurons_atframes, subset_stim = oneshot_utils.get_neurons_atframes(timeline,spks, bin=4)
    else:
        neurons_atframes, subset_stim = oneshot_utils.get_neurons_atframes(timeline,spks)
    avg_response, csig = oneshot_utils.get_tuned_neurons(neurons_atframes, subset_stim)
    return neurons_atframes, xpos, ypos, iplane, subset_stim, csig

def load_neurons_stims_nat30k(root, session='TX68_12_28_21', highres = True, highvar = True):
    """ load 30k stimulus data """
    img_downsample = 1 if highres else 3

    # load images
    img, labels = load_images_mat(root, file='nat30k.mat', downsample=img_downsample, normalize=True)

    # load neurons
    dat = np.load(os.path.join(root, f'{session}_nat30k.npz'), allow_pickle=True) # neural data
    spks = dat['S']
    ypos, xpos = dat['ypos'], dat['xpos']

    from scipy.stats import zscore
    spks = spks.T
    spks_mean = spks.mean(axis=0)
    spks -= spks_mean
    spks /= spks.std(axis=0)

    if highvar:
        igood = dat['csig'] > dat['csig'].mean()
        spks = spks[:, igood]
    else:
        igood = np.ones(len(dat['csig']), dtype=bool)
    sigvar = dat['csig'][igood]
    print(f'mean signal variance of chosen neurons: {sigvar.mean():.2f}')
        
    # istim are indices of presented stims
    istim = dat['istim'].astype('int') - 1
    img = img[istim]
    n_stim, Ly, Lx = img.shape
    print(n_stim, Ly, Lx)
    n_stim, n_neurons = spks.shape

    ### divide data

    # (IMPORTANT) put all repeated data into training set
    # test and val should be unseen stimuli
    ist, reps = np.unique(istim, return_counts=True)
    ireps = np.isin(istim, ist[reps > 1])
    n_reps = ireps.sum()

    seed = 0
    train_frac = 0.8 - (n_reps / n_stim)
    np.random.seed(seed)
    itrain = np.random.rand(n_stim - n_reps) < train_frac
    itest = ~itrain
    itest[itest>0] = np.random.rand(itest.sum()) > 0.5
    ival = np.logical_and(~itrain, ~itest)

    inds = [itrain, ival, itest]
    inds0 = []
    for ind in inds:
        ind0 = np.zeros(n_stim, 'bool')
        ind0[~ireps] = ind
        inds0.append(ind0)
    itrain, ival, itest = inds0
    itrain[ireps] = True

    return img, spks, itrain, ival, itest, sigvar, igood


def load_txt32_val(crop=True, downsample=2):
    """ load texture 32x500 dataset for validating the classification performance of predicted activity from the encoding model."""
    data_root = '/home/carsen/dm11_pachitariu'
    data_path = os.path.join(data_root, 'data/STIM/text32_500.mat')
    stims = io.loadmat(data_path, squeeze_me=True)
    img = stims['img'] # (150, 400, 500, 32)
    img = img.transpose(0, 1, 3, 2) # (150, 400, 32, 500)
    img = img.reshape(150, 400, -1)
    print(f'load {data_path} done!')

    # images that were shown
    Ly, Lx, nf = img.shape # (150, 400, 16000)
    print('original img: ', img.shape, img.min(), img.max())

    img = img.astype(np.float32)
    timg = np.transpose(img, (2, 0, 1)) # (16000, 150, 400)
    timg = np.array([cv2.resize(im, (176, 66)) for im in timg]) # (16000, 66, 176)
    if crop:
        timg = timg[:,:,int(30) : int(150)] # (16000, 66, 120)
        timg = np.array([cv2.resize(im, (int(120/downsample), int(66/downsample))) for im in timg]) # (16000, 33, 60)
    else:
        timg = np.array([cv2.resize(im, (int(176/downsample), int(66/downsample))) for im in timg])
    timg -= timg.mean()
    timg /= timg.std()
    print('img:', timg.shape, timg.min(), timg.max())

    n_samples = 16000
    n_train = 8000
    n_test = 8000
    itrain = np.random.choice(np.arange(n_samples), n_train, replace=False)
    # use the rest for testing
    itest = np.setdiff1d(np.arange(n_samples), itrain)
    print(itrain.shape, itrain.min(), itrain.max())
    print(itest.shape, itest.min(), itest.max())

    label = [i//500 for i in range(32*500)]
    label = np.array(label)
    print('label: ', label.shape, label.min(), label.max())
    return timg, label, itrain, itest


def load_txt32(root, input_size=[66, 84], inds=None, normalize=False, only_behav=False):
    """ load images from Farah's 32 texture, 500 exemplars """
    stims = loadmat(os.path.join(root + 'text32_500.mat'), squeeze_me=True)
    if only_behav:
        cat_idx = [3, 24, 6, 12, 29, 30, 13, 15]
        img_idx = [434, 42]
        only_behav_cats = stims['img'][:,:,:,cat_idx]
        behav_mat = only_behav_cats[:,:,img_idx,:]
        behav_mat[:,:,1,3] = stims['img'][:,:,199,12]
        imgs = behav_mat.transpose(0,1,3,2).reshape(150, 400, -1)
        n_classes = 8
    else:
        imgs = stims['img'].transpose(0,1,3,2).reshape(150, 400, -1)
        n_classes = 32
    timgs = imgs.transpose(2,0,1)
    if input_size is not None:
        #timgs = np.array([cv2.resize(img[:, 30:150], input_size[::-1]) for img in timgs]).astype('float32')
        timgs = np.array([cv2.resize(img, input_size[::-1]) for img in timgs]).astype('float32')
    n_stims, Ly, Lx = timgs.shape
    n_reps = n_stims // n_classes
    print(n_stims,n_classes,n_reps)
    labels = np.tile(np.arange(0, n_classes)[:,np.newaxis], (1, n_reps)).flatten()

    if inds is not None:
        n_classes = len(inds)
        inds = (inds[:,np.newaxis] * n_reps + np.arange(0, n_reps)).flatten()
        timgs = timgs[inds]
        labels = np.tile(np.arange(0, n_classes)[:,np.newaxis], (1, n_reps)).flatten()

    # normalize images
    if normalize:
        timgs -= timgs.mean()
        timgs /= timgs.std()

    return timgs, labels

def load_behavtextures(root, input_size=[66, 84], normalize=False):
    stims = loadmat(os.path.join(root + '8x2Behav.mat'), squeeze_me=True)
    timgs = stims['img'].transpose(2,0,1)
    n_classes = 8
    if input_size is not None:
        #timgs = np.array([cv2.resize(img[:, 30:150], input_size[::-1]) for img in timgs]).astype('float32')
        timgs = np.array([cv2.resize(img, input_size[::-1]) for img in timgs]).astype('float32')
    n_stims, Ly, Lx = timgs.shape
    n_reps = n_stims // n_classes
    print(n_stims,n_classes,n_reps)
    labels = np.tile(np.arange(0, n_classes)[:,np.newaxis], (1, n_reps)).flatten()

    # normalize images
    if normalize:
        timgs -= timgs.mean()
        timgs /= timgs.std()

    return timgs.astype('float32'), labels

def load_8x4(root, input_size=[66, 84], normalize=False):
    stims = loadmat(os.path.join(root + '8x4textures.mat'), squeeze_me=True)
    timgs = stims['img'].transpose(2,0,1)
    n_classes = 8
    if input_size is not None:
        #timgs = np.array([cv2.resize(img[:, 30:150], input_size[::-1]) for img in timgs]).astype('float32')
        timgs = np.array([cv2.resize(img, input_size[::-1]) for img in timgs]).astype('float32')
    n_stims, Ly, Lx = timgs.shape
    n_reps = n_stims // n_classes
    print(n_stims,n_classes,n_reps)
    labels = np.tile(np.arange(0, n_classes)[:,np.newaxis], (1, n_reps)).flatten()

    # normalize images
    if normalize:
        timgs -= timgs.mean()
        timgs /= timgs.std()

    return timgs.astype('float32'), labels


def load_one_shot(root, twoclass=False, input_size=[66, 84]):
    """ load images for one shot learning from Miguel's task """
    fname = os.path.join(root, 'miguel_passive2.mat') if twoclass else os.path.join(root, 'miguel_passive8x4.mat')
    dstim = loadmat(fname, squeeze_me=True)
    imgs = dstim['img'].transpose(2,0,1)

    if not twoclass:
        imgs = imgs[-80:]
        timgs = np.array([cv2.resize(img[:, 30:150], input_size[::-1]) for img in imgs])
        ### data divisions
        n_reps = imgs.shape[0] // 8
        n_stims = imgs.shape[0]
        labels = np.tile(np.arange(n_stims//n_reps)[:,np.newaxis], (1,n_reps)).flatten()
    else:
        timgs = np.array([cv2.resize(img[:, 30:150], input_size[::-1]) for img in imgs])
        timgs = np.stack((timgs, 
                    np.array([cv2.resize(img[:, 150:270], input_size[::-1]) for img in imgs])),
                    axis=0)
        timgs = timgs.transpose(1,0,2,3).reshape(-1, *input_size)
        ### data divisions
        n_stims = imgs.shape[0]
        n_classes = 2
        labels = np.zeros(n_stims, 'int')
        labels[n_stims//n_classes:] = 1

    timgs = timgs.astype('float32')

    return timgs, labels


def load_images_mat(root, file='nat30k.mat', downsample=3, normalize=True, crop=True, return_txt_label=False, origin=False):
    """ load images from mat file """
    path = os.path.join(root, file)
    dstim = loadmat(path, squeeze_me=True) # stimulus data
    labels = dstim['istim']
    img = np.transpose(dstim['img'], (2,0,1)).astype('float32')
    n_stim, Ly, Lx = img.shape
    print('nat30k raw image shape: ', img.shape)
    
    img = np.array([cv2.resize(im, (int(Lx//downsample), int(Ly//downsample))) for im in img])

    # crop image based on RF locations
    if origin:
        img = img
    elif crop:
        img = img[:,:,int(30//downsample) : int(150//downsample)]
    else:
        img = img[:,:,: int(176//downsample)] # keep left and middle screen
    print('nat30k image mean: ', img.mean())
    print('nat30k image std: ', img.std())
    # nat30k image mean:  127.59005
    # nat30k image std:  57.503426
    
    # normalize images
    if normalize:
        img -= img.mean()
        img /= img.std()
    if return_txt_label:
        return img, labels, dstim['iorig']
    else:
        return img, labels

def load_dataloaders(root, folder_name, seed=0, batch_size=64):
    try:
        from lurz2020.datasets.mouse_loaders import static_loaders
    except Exception as e:
        print('pip install git+https://github.com/sinzlab/Lurz_2020_code.git')
        raise ImportError(e)
    
    paths = [os.path.join(root, folder_name)]

    dataset_config = {'paths': paths, 
                      'batch_size': batch_size, 
                      'seed': seed, 
                      'cuda': True,
                      'normalize': True, 
                      'exclude': 'images'} # exclude images from normalization (default behavior)

    dataloaders = static_loaders(**dataset_config)
    
    return dataloaders

def create_lurz_dataloaders(X, ypos, xpos, img, root, folder_name, train_frac=0.8, seed=0, batch_size=64):
    """ create specialized torch dataloaders for lurz encoding model 
    
    X : (n_stim, n_neurons) responses of neurons
    
    ypos : (n_neurons,) y-position of neurons
    
    xpos : (n_neurons,) x-position of neurons
    
    img : (n_stim, Ly, Lx)
    
    root : str, save path
    
    folder_name : str, folder in save path to create
    
    train_frac : float, fraction of images to use for training
    
    seed : int, random seed for train / test / val split and model
    
    batch_size : int, batch size for dataloader
    
    """
    try:
        from lurz2020.datasets.mouse_loaders import static_loaders
    except Exception as e:
        print('pip install git+https://github.com/sinzlab/Lurz_2020_code.git')
        raise ImportError(e)
    
    n_stim, n_neurons = X.shape

    if X.shape[0] != img.shape[0]:
        raise ValueError('X.shape[0] != img.shape[0], require same number of stimuli in X and img')

    img_stats = [img.max(), img.min(), img.mean(), np.median(img[:]), img.std()]
    img_stats_folder = os.path.join(root, folder_name, 'meta', 'statistics', 'images')
    os.makedirs(img_stats_folder, exist_ok=True)

    X_stats = [X.max(axis=0), X.min(axis=0), X.mean(axis=0), np.median(X, axis=0), X.std(axis=0)]
    X_stats_folder = os.path.join(root, folder_name, 'meta', 'statistics', 'responses')
    os.makedirs(X_stats_folder, exist_ok=True)

    stats_names = ['max', 'min', 'mean', 'median', 'std']

    for folder in ['all', 'stimulus_frame']:
        Xs_folder = os.path.join(X_stats_folder, folder)
        os.makedirs(Xs_folder, exist_ok=True)
        imgs_folder = os.path.join(img_stats_folder, folder)
        os.makedirs(imgs_folder, exist_ok=True)
        for istat, xstat, sname in zip(img_stats, X_stats, stats_names):
            np.save(os.path.join(Xs_folder, f'{sname}.npy'), xstat)
            np.save(os.path.join(imgs_folder, f'{sname}.npy'), istat)

    img_folder = os.path.join(root, folder_name, 'data', 'images')
    os.makedirs(img_folder, exist_ok=True)
    for i in range(len(img)):
        np.save(os.path.join(img_folder, f'{i}.npy'), img[[i], :, :])

    X_folder = os.path.join(root, folder_name, 'data', 'responses')
    os.makedirs(X_folder, exist_ok=True)
    for i in range(len(X)):
        np.save(os.path.join(X_folder, f'{i}.npy'), X[i])

    meta_neurons_folder = os.path.join(root, folder_name, 'meta', 'neurons')
    os.makedirs(meta_neurons_folder, exist_ok=True)

    np.save(os.path.join(meta_neurons_folder, 'area.npy'), np.ones(n_neurons, 'int'))
    np.save(os.path.join(meta_neurons_folder, 'cell_motor_coordinates.npy'),
            np.stack((ypos, xpos, np.zeros_like(ypos)), axis=1))
    np.save(os.path.join(meta_neurons_folder, 'animal_ids.npy'), np.ones(n_neurons, 'int'))

    meta_trials_folder = os.path.join(root, folder_name, 'meta', 'trials')
    os.makedirs(meta_trials_folder, exist_ok=True)

    np.random.seed(seed)
    itrain = np.random.rand(n_stim) < train_frac
    itest = ~itrain
    itest[itest>0] = np.random.rand(itest.sum()) > 0.5
    ival = np.logical_and(~itrain, ~itest)
    inds = [itrain, ival, itest]

    tiers = np.zeros(n_stim, object)
    tiers[itrain] = 'train'
    tiers[ival] = 'validation'
    tiers[itest] = 'test'
    tiers = np.array(list(tiers))

    np.save(os.path.join(meta_trials_folder, 'tiers.npy'), tiers)
    np.save(os.path.join(meta_trials_folder, 'frame_image_id.npy'), np.arange(0, n_stim))
    
    paths = [os.path.join(root, folder_name)]

    dataset_config = {'paths': paths, 
                      'batch_size': batch_size, 
                      'seed': seed, 
                      'cuda': True,
                      'normalize': True, 
                      'exclude': 'images'} # exclude images from normalization (default behavior)

    dataloaders = static_loaders(**dataset_config)
    
    return dataloaders

def random_rotate_and_resize(X, scale_range=1.0, xy = (224,224), do_flip=True):
    """ augmentation by random rotation and resizing

        X is array with dims channels x Ly x Lx (channels optional)

        Parameters
        ----------
        X: list of ND-arrays, float 
            list of image arrays of size [nchan x Ly x Lx] or [Ly x Lx]

        scale_range: float (optional, default 1.0)
            Range of resizing of images for augmentation. Images are resized by 
            1.0 + scale_range * np.random.rand()
        
        xy: tuple, int (optional, default (224,224))
            size of transformed images to return

        do_flip: bool
            whether or not to flip images horizontally

        Returns
        -------
        imgi: ND-array, float
            transformed images in array [nimg x nchan x xy[0] x xy[1]]

        scale: array, float
            amount each image was resized by
            
        theta: array, float
            rotation angle

    """
    scale_range = max(0, min(2, float(scale_range)))
    
    Ly, Lx = X.shape[-2:]

    # generate random augmentation parameters
    flip = np.random.rand()>.5
    theta = np.random.rand() * np.pi * 2
    scale = 1.0 + scale_range * np.random.rand()
    dxy = np.maximum(0, np.array([Lx*scale-xy[1],Ly*scale-xy[0]]))
    dxy = (np.random.rand(2,) - .5) * dxy

    # create affine transform
    cc = np.array([Lx/2, Ly/2])
    cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
    pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
    pts2 = np.float32([cc1,
            cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
            cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
    M = cv2.getAffineTransform(pts1,pts2)

    img = X.copy()
        
    if flip and do_flip:
        img = img[:, :, ::-1]
        
    nchan = X.shape[0]
    imgi = np.zeros((nchan, xy[0], xy[1]), np.float32)
    for k in range(nchan):
        imgi[k] = cv2.warpAffine(img[k], M, (xy[1],xy[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) # edit by Farah

    return imgi, scale, theta


class TextureDataset(Dataset):
    def __init__(self, root_file, split_fraction=0., bin_size=2, 
                  n_channels=1, exclude_inds=None):
        img = np.load(root_file)
        print('file loaded')
        self.bin_size = bin_size
        nimg,Ly,Lx = img.shape
        if exclude_inds is not None:
            inds = np.ones(nimg, 'bool')
            inds[exclude_inds] = False 
            img = img[inds]    
        img = img[:, :bin_size*(Ly//bin_size), :bin_size*(Lx//bin_size)]
        img = img.reshape(-1, Ly//bin_size, bin_size, 
                          Lx//bin_size, bin_size).sum(axis=-1).sum(axis=-2) / bin_size**2
        self.images = img[:,np.newaxis]
        
        if n_channels > 1:
            self.images = np.tile(self.images, (1, n_channels, 1, 1))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = self.images[idx]
        
        image = self.transform(image)
        sample = {'image': image, 'label': idx}

        return sample
    
    def transform(self, image):
        return random_rotate_and_resize(image, xy=(112*2//self.bin_size, 112*2//self.bin_size))[0]
    
