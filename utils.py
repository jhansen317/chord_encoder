import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import numpy as np
from matplotlib.gridspec import GridSpec




ORIGINAL_DIM = 128
LATENT_DIM = 2
MIN_MIDI = 30
MAX_MIDI = 90
DATA_LENGTH = MAX_MIDI - MIN_MIDI

device = "cpu" # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def denseMidi2Multihot(dense_midis, width=ORIGINAL_DIM):
    chord_zeroes = torch.zeros(ORIGINAL_DIM,dtype=torch.float)
    chord_zeroes[dense_midis] = 1.
    dense = torch.tensor(midis, dtype=torch.float)
    all_mh = torch.cat([all_mh, torch.unsqueeze(chord_zeroes, dim=0)], dim=0)
    all_dense = torch.cat([all_dense, torch.unsqueeze(dense, dim=0)], dim=0)


def raggedMidis2Dense(ragged_midis):
    if not torch.is_tensor(ragged_midis):
        chords_ragged = [torch.tensor(m) for m in ragged_midis]
        return pad_sequence(chords_ragged)
    else:
        return ragged_midis.detach()
    

def midi2tensor(midis, width=ORIGINAL_DIM):
    chords_square = raggedMidis2Dense(midis)
    num_chords = len(midis)
    src =torch.ones([num_chords, width])
    zeros = torch.zeros([num_chords, width], dtype=src.dtype)
    #return torch.unsqueeze(zeros.scatter_(1, chord, src), 1)
    return zeros.scatter_(1, chords_square, src)

def getTopK(multi_hots, k):
    return torch.topk(multi_hots, k).indices

def getTopKAsMultiHot(multi_hots, k):
    return midi2tensor(getTopK(multi_hots, k))

def tensor2midi(multi_hots, offset=0):
    multi_hots = torch.squeeze(multi_hots)
    denses = []
    for mh in multi_hots:
        raw = torch.squeeze(torch.nonzero(mh), 1)
        denses.append(raw)
    return torch.transpose(pad_sequence(denses), 1,0)

def midi2ohe(midis, offset=0, width=ORIGINAL_DIM):
    return F.one_hot(torch.tensor(midis), width).to(torch.float)

def ohe2midi(one_hots, offset=0):
    return torch.argmax(one_hots, dim=-1)

# for saving the reconstructed images
def save_decoded_image(img, name):
    dense = ohe2midi(img)
    mhe = midi2tensor(dense)
    mhe = mhe.view(mhe.size(0), 1, ORIGINAL_DIM, 1)
    save_image(img, name, nrow=128)

def save_interleaved(img, outputs, name, input_encoding='midi'):
    bl = outputs.size(0)
    mh_outputs = torch.squeeze(outputs) 
    mh_outputs = torch.repeat_interleave(mh_outputs, 8, dim=1)
    mh_outputs = torch.repeat_interleave(mh_outputs, 8, dim=-1)

    outputs = mh_outputs.view(bl, 1, DATA_LENGTH*8, 8).cpu().data
    mh_imgs = torch.squeeze(img) 
    mh_imgs = torch.repeat_interleave(mh_imgs, 8, dim=1)
    mh_imgs = torch.repeat_interleave(mh_imgs, 8, dim=-1)
    imgs = mh_imgs.view(bl, 1, DATA_LENGTH*8, 8).cpu().data
    out = torch.cat((outputs, imgs), -1)
    save_image(out, name, nrow=bl // 2)

class ChoraleChords(Dataset):
    def __init__(self, encoding='multi_hot', augmented=True):
        self.all_mh = []
        self.augmented = augmented
        self.encoding = encoding
        self.original_dense = torch.load('original_dense.pt').unique(dim=0)
        self.transposed_dense = torch.load('all_dense.pt').unique(dim=0)
        self.transposed_mh= torch.load('all_multihot.pt').unique(dim=0)
        self.original_mh  = torch.load('original_multihot.pt').unique(dim=0)
        if self.augmented:
            self.length = len(self.transposed_mh) 
        else:
            self.length = len(self.original_mh)
        
    def __len__(self):
        return self.length

    
    def __getitem__(self, idx):
        if self.encoding == 'multi_hot':
            if self.augmented:
                return self.transposed_mh[idx]
            else:
                return self.original_mh[idx]
        else:
            if self.augmented:
                return self.transposed_dense[idx]
            else:
                return self.original_dense[idx]

def plot_embedding_space(embeddings, colors, epoch=0, loss=0.0):
    #fig, ax = plt.subplots(figsize=(12, 10))
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 10))
    ax1.scatter(embeddings[:, 0], embeddings[:, 1], s=0.5, alpha=1, c=colors)
    #ax1.annotate(f"epoch: {epoch}, loss: {loss}", [-4.5,-4.5])
    fig.suptitle(f"Chord Embeddings + examples, epoch: {epoch}, loss: {loss}")

    fig.tight_layout()
    #print(f'{datetime.now():%Y-%m-%d %H:00:00}')
    fig.savefig(f"./outputs/images/embedding_space_{epoch}.png")
    plt.close()

def plot_interleaved(img, outputs,embeddings, colors, epoch=0, loss=0.0):
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(12, 10))
    bl = outputs.size(0)
    mh_outputs = torch.squeeze(outputs) # midi2tensor(torch.squeeze(dense_outputs, 1))
    mh_outputs = torch.repeat_interleave(mh_outputs, 8, dim=1)
    mh_outputs = torch.repeat_interleave(mh_outputs, 8, dim=-1)

    outputs = mh_outputs.view(bl, 1, DATA_LENGTH*8, 8).cpu().data
    mh_imgs = torch.squeeze(img) #midi2tensor(dense_imgs)
    mh_imgs = torch.repeat_interleave(mh_imgs, 8, dim=1)
    mh_imgs = torch.repeat_interleave(mh_imgs, 8, dim=-1)
    imgs = mh_imgs.view(bl, 1, DATA_LENGTH*8, 8).cpu().data
    out = torch.cat((outputs, imgs), -1)
    ax2.imshow(out, cmap='gray', aspect='auto', interpolation='nearest')
    fig.tight_layout()
    fig.savefig(f"./outputs/images/example_images_{epoch}.png")
    plt.close()
    #save_image(out, name, nrow=bl // 2)


'''
config = {
    'original_dim': ORIGINAL_DIM,
    'intermediate_dim': INTERMEDIATE_DIMS,
    'latent_dim': LATENT_DIM,
    'data_length': DATA_LENGTH,
    'learning_rate': LEARNING_RATE,
    'kld_weight': KLD_WEIGHT,
    'bce_weight': WEIGHT_MULT,
    'class_weight_exponent': 1
}

'''
def plot_details(img, outputs,embeddings, 
                 colors, epoch=0, epoch_ind = 0, 
                 filename=None, weighted_losses=None, 
                 unweighted_losses=None, 
                 sample_targets=None,
                 sample_outputs=None,
                 sample_embeddings=None,
                **config):
    #fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(24, 12))
    #fig, axs = plt.subplots(2, 2,figsize=(24, 12))
    fig = plt.figure(figsize=(50, 25), layout="constrained")
    gs = GridSpec(12, 8, figure=fig)
    textstr = '\n'.join((
    f"Data length: {config['data_length']}",
    f"Intermediate Dims: {config['intermediate_dim']}",
    f"Latent Dims: {config['latent_dim']}",
    f"Learning Rate: {config['learning_rate']}",
    f"BCE Loss Weight: {config['bce_weight']}",
    f"KL Loss Weight: {config['kld_weight']}",
    f"Softmax Dim for loss: {config['kl_softmax_dim']}",
    f"Inverse Frequency Weighted Loss: {config['weight_inverse_frequency']}"
    ))


    embedding_ax = fig.add_subplot(gs[0:10, 2:6], title='embedding space')
    recons_ax = fig.add_subplot(gs[0:6, 0:2], title='reconstructions')
    targets_ax = fig.add_subplot(gs[6:12, 0:2], title='targets')
    #pitch_weights_ax = fig.add_subplot(gs[5:9, 10:11], sharey=targets_ax, xmargin=0, ymargin=0)
    #chord_weights_ax = fig.add_subplot(gs[9:10, 10:20], sharex=targets_ax, xmargin=0, ymargin=0)
    sample_targets_ax = fig.add_subplot(gs[10:12,  2:6])
    #samples_ax = fig.add_subplot(gs[6:9, 0:2])
    weighted_bces_ax = fig.add_subplot(gs[0:3, 6:8], title='weighted BCE loss')
    weighted_kls_ax = fig.add_subplot(gs[3:6, 6:8], title='weighted KL loss')
    unweighted_bces_ax = fig.add_subplot(gs[6:9, 6:8], title='unweighted BCE loss')
    unweighted_kls_ax = fig.add_subplot(gs[9:12, 6:8], title='unweighted KL loss')


    #unweighted_losses_ax = fig.add_subplot(gs[2:4, 6:8])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    embedding_ax.text(0.05, 0.95, textstr, transform=embedding_ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    targets_ax.tick_params(bottom = False,left=True, labelbottom=False,  labelleft=True)
    recons_ax.tick_params(bottom = False,left=True, labelbottom=False,  labelleft=True)
    #pitch_weights_ax.tick_params(bottom = False,left=True, top=True, labeltop=True, labelbottom=False,  labelleft=True)
    #chord_weights_ax.tick_params(bottom = True,left=False, top=False, labeltop=False, labelbottom=True,  labelleft=False, right=True, labelright=True)
    #ax3 = fig.add_subplot(gs[3, :])
    bl = outputs.size(0)
    #print(unweighted_losses.shape)
    #print(weighted_losses.shape)
    losses = torch.squeeze(unweighted_losses[:, epoch_ind])
    wlosses = torch.squeeze(weighted_losses[:, epoch_ind])
    unweighted_bces = unweighted_losses[0, :]
    unweighted_kls = unweighted_losses[1, :]

    weighted_bces = weighted_losses[0, :]
    weighted_kls = weighted_losses[1, :]
    unweighted_bces = unweighted_bces.T
    unweighted_kls = unweighted_kls.T
    weighted_bces = weighted_bces.T
    weighted_kls = weighted_kls.T


    #ax1.annotate(f"epoch: {epoch}, loss: {loss}", [-4.5,-4.5])
    suptitle = '\n'.join((
        f"Epoch {epoch}",
        f"Weighted Cross Entropy Loss {wlosses[0]:.5f}",
        f"UnWeighted Binary Cross Entropy Loss {losses[0]:.5f}",
        f"Weighted KL loss: {wlosses[1]:.5f}",
        f"UnWeighted KL loss: {losses[1]:.5f}"
        ))
    
    fig.suptitle(suptitle, fontsize=14)
    #print(colors.shape)
    out = torch.squeeze(outputs)
    #img = torch.flip(img,[1,0])
    print(torch.nonzero(sample_targets)[:, 1])

    smin =  torch.squeeze(torch.min(torch.nonzero(sample_targets)[:, 1])).tolist()
    smax = torch.squeeze(torch.max(torch.nonzero(sample_targets)[:, 1])).tolist()
    sample_recons = sample_outputs
    #sample_recons = getTopKAsMultiHot(sample_recons, 4)
    sample_img = sample_targets
    recons_t = torch.tensor([])
    for i in range(0, len(sample_recons)):
        recons_t = torch.cat([recons_t, torch.unsqueeze(torch.zeros_like(sample_recons[i]), dim=0).detach()], dim=0)
        recons_t = torch.cat([recons_t, torch.unsqueeze(sample_img[i], dim=0).detach()], dim=0)
        recons_t = torch.cat([recons_t, torch.unsqueeze(sample_recons[i], dim=0).detach()], dim=0)
        recons_t = torch.cat([recons_t, torch.unsqueeze(torch.zeros_like(sample_recons[i]), dim=0).detach()], dim=0)


    img = torch.transpose(img,1,0)
    #out = torch.flip(out,[1,0])
    out = torch.transpose(out,1,0)
    #sample_recons = torch.transpose(getTopKAsMultiHot(sample_recons, 4),1,0)
    sample_img = torch.transpose(recons_t,1,0)
    out = out.view(DATA_LENGTH, bl)
    img = img.view(DATA_LENGTH, bl)
    embedding_ax.scatter(embeddings[:, 0], embeddings[:, 1], s=4, alpha=1, c=colors)
    embedding_ax.scatter(sample_embeddings[:, 0], sample_embeddings[:, 1], s=40, alpha=0.7, c=torch.ones(len(sample_embeddings))*1280)


    recons_ax.imshow(out.detach().numpy(), cmap='gray', aspect='auto', interpolation='nearest')
    targets_ax.imshow(img.detach().numpy(), cmap='gray', aspect='auto', interpolation='nearest')
    sample_targets_ax.imshow(sample_img.detach().numpy(), cmap='gray', aspect='auto', interpolation='nearest')
    #samples_ax.imshow(sample_recons.detach().numpy(), cmap='gray', aspect='auto', interpolation='nearest')
    
    #targets_ax.tick_params(bottom = False)
    #pitch_weights_ax.barh(range(len(pitchwise_weights)), pitchwise_weights)
    #chord_weights_ax.bar(range(len(chordwise_weights)), chordwise_weights)
    weighted_bces_ax.plot(weighted_bces.detach().numpy())
    weighted_kls_ax.plot(weighted_kls.detach().numpy())
    unweighted_bces_ax.plot(unweighted_bces.detach().numpy())
    unweighted_kls_ax.plot(unweighted_kls.detach().numpy())
    if filename is not None:
        fig.savefig(filename)
    plt.close()