import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os
import json
from VAEModels import GaussianVAE
import music21



ORIGINAL_DIM = 128
LATENT_DIM = 2
MIN_MIDI = 0
MAX_MIDI = 128
MIDI_WIDTH = 128
DURATION_WIDTH = 11  # 14359
NORMALS_WIDTH = 11
DATA_LENGTH = MAX_MIDI - MIN_MIDI + NORMALS_WIDTH  # + DURATION_WIDTH

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def denseMidi2Multihot(dense_midis, width=ORIGINAL_DIM):
    chord_zeroes = torch.zeros(ORIGINAL_DIM, dtype=torch.float)
    chord_zeroes[dense_midis] = 1.0
    dense = torch.tensor(midis, dtype=torch.float)
    all_mh = torch.cat([all_mh, torch.unsqueeze(chord_zeroes, dim=0)], dim=0)
    all_dense = torch.cat([all_dense, torch.unsqueeze(dense, dim=0)], dim=0)

def midi_to_padded(all_midis, max_chord_length=25):
    set_size = len(all_midis)
    nested_midis = torch.nested.nested_tensor(
        all_midis, dtype=torch.int64, requires_grad=False, pin_memory=True
    )
    return torch.nested.to_padded_tensor(
        nested_midis, 0.0, output_size=[set_size, max_chord_length]
    ).to(torch.int64).detach()


def midi_to_tensor(all_midis, max_chord_length=25):
    set_size = len(all_midis)
    if max_chord_length == 0:
        chord_lengths = [len(list(c)) for c in all_midis]
        max_chord_size = max(chord_lengths)
    else:
        max_chord_size = max_chord_length
    mh_zeros = torch.zeros([set_size, 129],dtype=torch.int64, requires_grad=False)
    mh_ones = torch.ones([set_size, 129],dtype=torch.int64, requires_grad=False)
    nested_midis = torch.nested.nested_tensor(
        list(all_midis), dtype=torch.int64, requires_grad=False, pin_memory=True
    )
    all_dense = torch.nested.to_padded_tensor(
        nested_midis, 0.0, output_size=[set_size, max_chord_size]
    ).to(torch.int64).detach()

    all_mh = mh_zeros.scatter_(1, all_dense, mh_ones)[:, 1:].detach()
    all_normals = [sorted(list({(m - midis[0]) % 12 for m in midis})) for midis in all_midis]
    normal_zeros = torch.zeros([set_size, 12])
    normal_ones = torch.ones([set_size, 12])
    nested_normals = torch.nested.nested_tensor(list(all_normals), dtype=torch.int64, requires_grad=False, pin_memory=True)
    normals_dense = torch.nested.to_padded_tensor(nested_normals, 0.0, output_size=[set_size, 12]).to(torch.int64).detach()
    normal_mh = normal_zeros.scatter_(1, normals_dense, normal_ones)[:, 1:].detach()
    all_masks = all_dense.eq(0.0).to(torch.float).detach()
    return all_mh, all_dense, normal_mh, all_normals, all_masks


def getTopK(multi_hots, k):
    return torch.topk(multi_hots, k).indices.tolist()


def getTopKAsMultiHot(multi_hots, k):
    return midi_to_tensor(getTopK(multi_hots[:, 0:128], k))


def tensor2midi(multi_hots, offset=0):
    multi_hots = multi_hots
    denses = []
    for mh in multi_hots:
        raw = torch.squeeze(torch.nonzero(mh), 1)
        denses.append(raw)
    return torch.transpose(pad_sequence(denses), 1, 0)


def midi2ohe(midis, offset=0, width=ORIGINAL_DIM):
    return F.one_hot(torch.tensor(midis), width).to(torch.float)


def ohe2midi(one_hots, offset=0):
    return torch.argmax(one_hots, dim=-1)


def plot_embeddings(embeddings, ax, **kwargs):
    xs = embeddings[:, 0]
    ys = embeddings[:, 1]
    zs = embeddings[:, 2]
    ax.scatter3D(xs, ys, zs, **kwargs)


def plot_piano_roll(data, ax, **kwargs):
    ax.imshow(data, **kwargs)


def plot_reconstructions_interleaved(
    targets, reconstructions, ax, mid_min, min_max, **kwargs
):
    zeros = np.expand_dims(np.zeros_like(targets[0, mid_min:min_max]), axis=0)
    recons_t = zeros
    for i in range(0, len(reconstructions)):
        recons_t = np.vstack([recons_t, zeros])
        recons_t = np.vstack(
            [recons_t, np.expand_dims(targets[i, mid_min:min_max], axis=0)]
        )
        recons_t = np.vstack(
            [recons_t, np.expand_dims(reconstructions[i, mid_min:min_max], axis=0)]
        )
        recons_t = np.vstack([recons_t, zeros])
    plot_piano_roll(recons_t.T, ax, **kwargs)


def plot_details(
    targets,
    outputs,
    embeddings,
    colors,
    epoch=0,
    epoch_ind=0,
    filename=None,
    weighted_losses=None,
    unweighted_losses=None,
    sample_targets=None,
    sample_outputs=None,
    sample_embeddings=None,
    sample_colors=None,
    **config,
):
    targets = torch.sum(targets.view(-1, 25, 128), 1).detach().cpu().numpy()
    outputs = torch.sum(outputs.view(-1, 25, 128), 1).detach().cpu().numpy()
    embeddings = embeddings.detach().cpu().numpy()
    sample_targets = torch.sum(sample_targets.view(-1, 25, 128), 1).detach().cpu().numpy()
    sample_outputs = torch.sum(sample_outputs.view(-1, 25, 128), 1).detach().cpu().numpy()
    sample_embeddings = sample_embeddings.detach().cpu().numpy()
    weighted_losses = weighted_losses.detach().cpu().numpy()
    unweighted_losses = unweighted_losses.detach().cpu().numpy()
    fig = plt.figure(figsize=(24, 15), layout="constrained")
    fig2, full_interleaved = plt.subplots(1, 1, figsize=(24, 12))
    gs = GridSpec(12, 8, figure=fig)
    textstr = "\n".join(
        (
            f"Latent Dims: {config['latent_dim']}",
            f"Learning Rate: {config['learning_rate']}",
            f"Dropout Rate: {config['dropout']}",
            f"Hidden Layers: {config['hidden_dims']}"
            f"Hidden Activation: {config['hidden_activation']}",
            f"BCE Loss Weight: {config['bce_weight']}",
            f"KL Loss Weight: {config['kld_weight']}",
            f"Inverse Frequency Weighted Loss: {config['weight_inverse_frequency']}",
        )
    )

    embedding_ax = fig.add_subplot(
        gs[0:10, 2:6], title="embedding space", projection="3d"
    )
    recons_ax = fig.add_subplot(gs[0:6, 0:2], title="reconstructions")
    targets_ax = fig.add_subplot(gs[6:12, 0:2], title="targets")
    sample_targets_ax = fig.add_subplot(gs[10:12, 2:6])
    weighted_bces_ax = fig.add_subplot(gs[0:6, 6:8], title="CrossEntropy Loss")
    weighted_kls_ax = fig.add_subplot(gs[6:12, 6:8], title="KL-Divergence loss")

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    embedding_ax.text(
        0.05,
        0.95,
        0.0,
        textstr,
        transform=embedding_ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )
    targets_ax.tick_params(bottom=False, left=True, labelbottom=False, labelleft=True)
    recons_ax.tick_params(bottom=False, left=True, labelbottom=False, labelleft=True)
    wlosses = weighted_losses[:, epoch_ind]
    weighted_bces = weighted_losses[0, :]
    weighted_kls = weighted_losses[1, :]
    weighted_bces = weighted_bces.T
    weighted_kls = weighted_kls.T

    suptitle = "\n".join(
        (
            f"Epoch {epoch}",
            f"CrossEntropy Loss {wlosses[-1]:.5f}",
            f"KL loss: {wlosses[-1]:.5f}",
        )
    )

    fig.suptitle(suptitle, fontsize=14)
    fig2.suptitle("Targets and reconstructions", fontsize=14)
    plot_reconstructions_interleaved(
        sample_targets,
        sample_outputs,
        sample_targets_ax,
        0,
        128,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    plot_reconstructions_interleaved(
        sample_targets,
        sample_outputs,
        full_interleaved,
        0,
        128,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    plot_embeddings(embeddings, embedding_ax, s=10, alpha=0.4, c=colors)
    plot_embeddings(sample_embeddings, embedding_ax, s=40, alpha=0.8, c=sample_colors)
    plot_piano_roll(
        outputs.T,
        recons_ax,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )
    plot_piano_roll(
        targets.T,
        targets_ax,
        cmap="gray",
        aspect="auto",
        interpolation="nearest",
        origin="lower",
    )

    weighted_bces_ax.plot(weighted_bces)
    weighted_kls_ax.plot(weighted_kls)
    if filename is not None:
        fig.savefig(filename)
    fig2.savefig(f"interleaved_{epoch}.png")
    plt.close()
    plt.close()

def save_model_state(epoch, config, model, run_id):
    paths = setup_paths(run_id)
    model_file = os.path.join(paths['model_dir'], f"{epoch}.pth")
    config_file = os.path.join(paths['parent_dir'], "config.json")

    if not os.path.exists(config_file):
        with open(config_file, "w+") as f:
            json.dump(config, f)
    if not os.path.exists(config_file):
        with open(config_file, "w+") as f:
            json.dump(config, f)
    torch.save(model.state_dict(), model_file)

def setup_paths(run_id):
    parent_dir = os.path.join(os.getcwd(), "outputs", "data", run_id)
    image_dir = os.path.join(parent_dir, "images")
    model_dir = os.path.join(parent_dir, "model_state")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    return {
        'parent_dir': parent_dir,
        'image_dir': image_dir,
        'model_dir': model_dir
    }


def load_model(run_id, epoch, device):
    paths = setup_paths(run_id)
    model_file = os.path.join(paths['model_dir'], f"{epoch}.pth")
    config_file = os.path.join(paths['parent_dir'], "config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)   
    model = GaussianVAE(**config).to(device)
    model.load_state_dict(torch.load(model_file,map_location=torch.device('mps')))
    model.eval()
    return model

def score_to_model_inputs(score_path):
    all_midis = []
    score = music21.converter.parse(score_path)
    for chord in score.chordify().recurse().getElementsByClass(["Chord", "Measure"]):
        if "Chord" in chord.classes:
            all_midis.append([p.midi for p in chord.pitches])
    mmh, md, _, nd, masks = midi_to_tensor(all_midis)
    return mmh, md, nd, masks