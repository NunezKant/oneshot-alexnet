from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import scipy.io

def classifiers(X, labels, itrain, itest, ipair=[0,1], regularizer=1.0, verbose=True):
    """ compute classification performance on labels for X """
    if ipair is not None:
        inds = np.logical_or(labels==ipair[0], labels==ipair[1])
    else:
        inds = np.ones(X.shape[0], 'bool')

    indtrain = np.nonzero(inds)[0][itrain[inds]]
    indtest = np.nonzero(inds)[0][itest[inds]]

    clf = LogisticRegression(random_state=0, 
                             penalty='l2', 
                             solver='liblinear',
                             C=regularizer, max_iter=200).fit(X[indtrain], labels[indtrain])
    accL = clf.score(X[indtest], labels[indtest])
    if verbose:
        print('Logistic Regression Test Accuracy: ', accL)
    
    clf = KNeighborsClassifier(n_neighbors=1, metric='cosine').fit(X[indtrain], labels[indtrain])
    accK = clf.score(X[indtest], labels[indtest])
    
    dist, idxs = clf.kneighbors(X[indtest], return_distance=True)

    if verbose:
        print('Nearest Neighbor Test Accuracy: ', accK)

    return accL, accK

def dprime(X, labels, ipair=[0,1], dp_threshold=0.7, plot=False):
    """ compute d-prime for X on two categories in labels """
    inds = np.logical_or(labels==ipair[0], labels==ipair[1])
    xp = X[inds].copy()
    lp = np.unique(labels[inds], return_inverse=True)[1]
    
    x0 = xp[lp==0]
    x1 = xp[lp==1]
    
    dp = (x0.mean(axis=0) - x1.mean(axis=0)) / (x0.mean(axis=0) + x1.mean(axis=0)) 
    print((dp>dp_threshold).mean(), (dp<-dp_threshold).mean())
    lv = xp[:, dp > dp_threshold].mean(axis=-1)
    cr = xp[:, dp < -dp_threshold].mean(axis=-1)
    
    if plot:
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        for i in range(2):
            plt.plot(lv[labels==i][:20])
        plt.title(f'(a) neurons w/ dp > {dp_threshold}')

        plt.subplot(1,3,2)
        for i in range(2):
            plt.plot(cr[labels==i][:20])
        plt.title(f'(b) neurons w/ dp < -{dp_threshold}')

        plt.subplot(1,3,3)
        for i in range(2):
            plt.plot(lv[labels==i][:20] - cr[labels==i][:20])
        plt.title('subtract a and b')

    return lv, cr

def get_dprime(X1, X2):
    """ 
    compute d-prime for each neuron between X1 and X2 
    X1, X2: (n_samples, n_neurons)
    dp: (n_neurons, )
    """
    dp = (X1.mean(axis=0) - X2.mean(axis=0)) / np.sqrt((X1.var(axis=0) + X2.var(axis=0))/2)
    return dp
#########################################################################################################
# txt32 classification accuracy                                                                         #
#########################################################################################################
# load images
def load_txt32_images_for_classification(data_root='/home/stringlab/dm11_pachitariu/data/STIM/'):
    data_path = os.path.join(data_root, 'text32_500.mat')
    stims = scipy.io.loadmat(data_path, squeeze_me=True)
    img = stims['img'] # (150, 400, 500, 32)
    img = img.transpose(0, 1, 3, 2) # (150, 400, 32, 500)
    img = img.reshape(150, 400, -1)
    print(f'load {data_path} done!')

    # images that were shown
    img = img.astype(np.float32)
    img -= img.mean()
    img /= img.std()
    timg = np.transpose(img, (2, 0, 1))
    timg = np.array([cv2.resize(im, (60, 33)) for im in timg])
    print(timg.shape, timg.min(), timg.max())
    return timg

# split train test
def split_train_test_txt32():
    n_samples = 16000
    n_train = int(n_samples * 0.8)
    itrain = np.random.choice(np.arange(n_samples), n_train, replace=False)
    # use the rest for testing
    itest = np.setdiff1d(np.arange(n_samples), itrain)
    # print(itrain.shape, itrain.min(), itrain.max())
    # print(itest.shape, itest.min(), itest.max())

    label = [i//500 for i in range(32*500)]
    label = np.array(label)
    # print(label.shape, label.min(), label.max())
    return itrain, itest, label

def metric_txt32_classification(n_layers=2, mouse_id=1, cortex_layer=2):
    # load model 
    timg = load_txt32_images_for_classification()
    itrain, itest, label = split_train_test_txt32()
    from approxineuro.models.encoding_model import load_readout_nat30k_model
    # n_layers = 2 # number of conv layers
    multikernel = False
    model, _ = load_readout_nat30k_model(n_layers, mouse_id, multikernel=multikernel, cortex_layer=cortex_layer, \
        l1_norm=True, l1_lambda=1.0, cls=False)

    # compute responses for model
    model.eval()
    X = model.responses(timg, core=False)
    X = X.reshape(X.shape[0], -1)
    print('feature: ', X.shape, X.max(), X.min())

    # classification      
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, penalty='l2', C=1).fit(X[itrain], label[itrain])
    acc = clf.score(X[itest], label[itest])
    print(f'{n_layers}-layer model, classification accuracy: {acc}')
    return acc

    # classification accuracy: 0.6521875
    # sigmoid classification accuracy: 0.5946875

#########################################################################################################
# txt32 one shot accuracy                                                                               #
#########################################################################################################
def metric_txt32_oneshot(data_root='/home/stringlab/dm11_pachitariu/data/STIM/', n_layers=2, mouse_id=1, cortex_layer=2, ntrain=1):
    '''
    ntrain: number of training images per class, for oneshot, set ntrain=1. the rest of images in each class are used for testing.
    '''
    # 1. load images 
    from approxineuro import datasets
    timgs, labels = datasets.load_txt32(data_root, input_size=[66, 120],
                                        inds = np.array([3, 24, 6, 12, 29, 30, 13, 15]))
    n_stims, Ly, Lx = timgs.shape
    print(timgs.shape, labels.shape)
    n_classes = labels.max() + 1
    n_reps = n_stims // n_classes

    timgs = timgs.astype(np.float32)
    img_mean = 127.59005
    img_std = 57.503426
    timgs -= img_mean
    timgs /= img_std
    timg = np.array([cv2.resize(im, (60, 33)) for im in timgs])
    print(timg.shape, timg.min(), timg.max())

    # 2. one shot                                                                  
    multikernel = False
    # n_layers = 1 # number of conv layers
    # mouse_id = 1
    # cortex_layer = 2

    accsL = np.nan * np.zeros((n_classes,n_classes))
    accsK = np.nan * np.zeros((n_classes, n_classes)) 

    from approxineuro.models.encoding_model import load_readout_nat30k_model
    model, model_path = load_readout_nat30k_model(n_layers, mouse_id, multikernel=multikernel, cortex_layer=cortex_layer, l1_norm=True, l1_lambda=1.0)

    model.eval()
    X = model.responses(timg)
    print(X.shape, X.max(), X.min())
    np.random.seed(0)
    # ntrain = 1
    ntest = 499

    print(f'>>> ntest = {ntest}')
    itrain = np.zeros(n_stims, 'bool')
    itest = np.zeros(n_stims, 'bool')
    for k in range(ntrain):
        itrain[k::n_reps] = True
    for k in range(ntest):
        itest[(k+ntrain)::n_reps] = True

    ### compute performance on texture classification
    for i0 in range(n_classes):
        for i1 in range(i0+1, n_classes):
            accsL[i0,i1], accsK[i0,i1] = classifiers(X, labels, itrain, itest, 
                                                                ipair=[i0, i1], verbose=False)

    print(f'logreg: {np.nanmean(accsL):.2f}, 1-NN: {np.nanmean(accsK):.2f}')
    return accsL, accsK

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


import torch
def model_performance(model, mouse_id = 0, cortex_layer = 2, n_layers = 2, device = 'cuda', crop=True, return_pred=False, img_downsample = 2 ):
    # based on model_prediction_results.ipynb
    mice_names = ['TX80', 'TX68', 'L1_A4']
    experiment_dates = {'TX80': '06_17_22', 'TX68': '12_28_21', 'L1_A4': '08_22_22'}
    mname = mice_names[mouse_id]
    expdate = experiment_dates[mice_names[mouse_id]]

    use_sensorium_normalization = True

    from approxineuro import datasets
    print('loading neural data...')
    datapath = '/home/stringlab/Desktop/github/approxineuro/datasets/nat30k/'
    spks, istim, xpos, ypos, csig = datasets.load_nat30k(datapath, mouse_name=mname, exp_date=expdate, ilayer=cortex_layer, normalize=False)
    print(spks.shape, spks.min(), spks.max())
    
    print('loading images...')
    img, labels = datasets.load_images_mat(datapath, file='nat30k.mat', downsample=img_downsample, normalize=True, crop=crop)

    # istim are indices of presented stims
    istim = istim.astype(int) 
    img = img[istim]
    n_stim, Ly, Lx = img.shape
    print(img.shape, img.max(), img.min())

    itrain, ival, itest = datasets.split_data_with_repeats(istim)

    print('normalizing neural data...')
    if use_sensorium_normalization:
        spks_std = spks[itrain].std(axis=0)
        thresh = 0.01*spks_std
        precision = np.ones_like(spks_std) * thresh
        idx = spks_std > thresh
        precision[idx] = spks_std[idx]
        spks = spks / precision
    else:
        spks = spks - spks[itrain].mean(0)
        spks = spks / spks[itrain].std(0)

    img_train = torch.from_numpy(img[itrain]).to(device).unsqueeze(1)
    img_val = torch.from_numpy(img[ival]).to(device).unsqueeze(1)
    # img_test = torch.from_numpy(img[itest]).unsqueeze(1)
        
    spks_train = torch.from_numpy(spks[itrain]).to(device)
    spks_val = torch.from_numpy(spks[ival]) 
    spks_test = torch.from_numpy(spks[itest]) 
    device = 'cuda:0'
    # model, model_save_path = load_readout_nat30k_model(n_layers, mouse_id, multikernel=multikernel, cortex_layer=cortex_layer, l1_norm=True, l1_lambda=1.0, cls=False, clamp=True)

    ### compute responses for model
    model.eval()
    X = model.responses(img[itest], core=False)
    print(X.shape, X.max(), X.min())
    X = X.reshape(X.shape[0], -1)
    print(X.shape)
    spks_test_gpu = spks_test.to(device)
    spks_test_pred = torch.from_numpy(X).to(device)

    varexp = ((spks_test_gpu - spks_test_pred)**2).sum(axis=0) 
    spks_test_gpu -= spks_test_gpu.mean(axis=0)
    varexp /= (spks_test_gpu**2).sum(axis=0)
    varexp = 1 - varexp

    spks_test_pred -= spks_test_pred.mean(axis=0)

    cc = (spks_test_gpu * spks_test_pred).mean(axis=0) 
    cc /= (spks_test_gpu**2).mean(axis=0)**0.5 * (spks_test_pred**2 + 1e-3).mean(axis=0)**0.5
    print(f'varexp = {varexp.mean():0.4f}, cc = {cc.mean():0.4f}')
    cc = cc.cpu().numpy()
    varexp = varexp.cpu().numpy()

    if return_pred:
        return cc, varexp, X, spks_test
    else:
        return cc, varexp

if __name__ == '__main__':
    # acc = metric_txt32_classification()
    accL, accK = metric_txt32_oneshot(n_layers=3, mouse_id=1, cortex_layer=2, ntrain=1)


