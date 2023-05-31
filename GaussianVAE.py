import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import math
import os
import glob
import pickle
import json
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np
from torchsummary import summary
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import uuid
import pathlib
import music21
import webdataset as wds

from torch.profiler import profile, record_function, ProfilerActivity

from utils import plot_details, midi_to_tensor, setup_paths, save_model_state, score_to_model_inputs


from VAEModels import GaussianVAE

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

with open("./duration_map.json", "r") as f:
    duration_map = json.load(f)
# 13k 430
ORIGINAL_DIM = 128
LATENT_DIM = 3
MIN_MIDI = 0
MAX_MIDI = 128
DURATION_WIDTH = 11  #  len(duration_map) # 14359
NORMALS_WIDTH = 11
DATA_LENGTH = MAX_MIDI - MIN_MIDI  # + NORMALS_WIDTH  # + DURATION_WIDTH
INTERMEDIATE_DIMS = ORIGINAL_DIM * 32
LEARNING_RATE = 1e-2
RHO = 0.0074
BETA = 4
KLD_WEIGHT = 0.0  #
WEIGHT_MULT = 1.0  # 1e3
SPARSE_WEIGHT = 1
RUN_UUID = str(uuid.uuid1())
BATCH_SIZE = 256
KL_SOFTMAX_DIM = 1
EPOCHS = 8000
START_EPOCH = 0


resume_run = None  # "07750700-e6c4-11ed-a95b-3af9d377b57f"
TRAIN_NUM = 5148875
if resume_run is not None:
    RUN_UUID = resume_run
    parent_dir = os.path.join(os.getcwd(), "outputs", "data", resume_run)
    model_dir = os.path.join(parent_dir, "model_state")
    image_dir = os.path.join(parent_dir, "images")
    image_glob = os.path.join(image_dir, "*.png")
    model_glob = os.path.join(model_dir, "*.pth")
    config_file = os.path.join(parent_dir, "config.json")
    run_path = f"./outputs/data/{resume_run}"
    images = glob.glob(image_glob)
    models = glob.glob(model_glob)
    latest_image = max(images, key=os.path.getctime)
    latest_model_state = max(models, key=os.path.getctime)
    print(f"latest model path is {latest_model_state}")
    START_EPOCH = int(os.path.splitext(os.path.basename(latest_image))[0]) + 1
    EPOCHS = START_EPOCH + EPOCHS
    with open(f"./outputs/data/{resume_run}/config.json") as f:
        config = json.load(f)

    model = GaussianVAE(**config).to(device)
    model.load_state_dict(torch.load(latest_model_state))
else:
    with open("./config.json", "r") as f:
        config = json.load(f)
    model = GaussianVAE(**config).to(device)

#summary(model, torch.zeros([1, 25, 128]), device=device)
# trainset = ChoraleChords(augmented=True)
# testset = ChoraleChords(augmented=False)


def get_unbatched_tuples_transposed(src):
    for sample in src:
        transposed_midis = []
        transposed_normals = []
        all_midis = [s[0] for s in sample[0]]
        all_normals = [list({n % 12 for n in s[1]}) for s in sample[0]]
        all_meta = [
            (json.dumps(s[1], separators=(",", ":")), s[2], s[3], s[4])
            for s in sample[0]
        ]

        for i in range(len(all_midis)):
            for transposition in range(-12, 12, 12):
                if (
                    min(all_midis[i]) + transposition > 0
                    and max(all_midis[i]) + transposition < 128
                ):
                    all_midis.append([n + transposition for n in all_midis[i]])
                    all_normals.append(all_normals[i])
                    all_meta.append(all_meta[i])

        all_mh, _,_,_ = midi_to_tensor(all_midis)

        for smh in all_mh:
            yield (smh, all_meta)


def get_unbatched_tuples(src):
    for sample in src:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
            all_midis = [sorted(s[0]) for s in sample[0]]
            all_normals = [sorted(list({n % 12 for n in s[1]})) for s in sample[0]]
            all_meta = [
                (json.dumps(s[1], separators=(",", ":")), s[2], s[3], s[4])
                for s in sample[0]
            ]
            all_mh, all_dense, normal_mh, normal_dense, all_masks_inputs = midi_to_tensor(all_midis)
            all_masks = torch.unsqueeze(all_masks_inputs, dim=-1)
            all_masks = all_masks.expand((-1,25,128))
            all_oh = F.one_hot(all_dense, 129).to(torch.float)[:, :, 1:]
        print(prof.key_averages(group_by_stack_n=1).table(sort_by="self_cpu_memory_usage", row_limit=10))
        for i in range(len(all_oh)):
            yield (all_oh[i], all_meta[i], all_masks[i], all_masks_inputs[i])
        


def get_unbatched_tuples_for_validation(src):
    # idx = torch.randint(10,(1,)).tolist()
    for sample in src:
        # print(sample)
        all_midis = [sorted(s[0]) for s in sample[0]]
        all_normals = [sorted(list({n % 12 for n in s[1]})) for s in sample[0]]
        all_meta = [
            (json.dumps(s[1], separators=(",", ":")), s[2]) for s in sample[0]
        ]
        all_mh, all_dense, normal_mh, normal_dense, all_masks = midi_to_tensor(all_midis)
        all_masks_inputs = torch.unsqueeze(all_masks, dim=-1)
        all_masks = all_masks_inputs.expand((-1,25,128))
        all_oh = F.one_hot(all_dense, 129).to(torch.float)[:, :, 1:]
        for i in range(len(all_oh)):
            yield (all_oh[i], all_meta[i], all_masks[i], all_masks_inputs[i])
        break



trainset = (
    wds.WebDataset("./shards_durations3/chordnet-train-{000000..000034}.tgz")
    .decode()
    .to_tuple("cbor")
    .compose(get_unbatched_tuples)
    .shuffle(TRAIN_NUM)
    .batched(config["batch_size"])
)
testset = (
    wds.WebDataset("./shards_durations3/chordnet-train-{000000..000034}.tgz")
    .decode()
    .to_tuple("cbor")
    .shuffle(5)
    .compose(get_unbatched_tuples_for_validation)
    .batched(config["batch_size"])
)


trainloader = DataLoader(trainset, batch_size=None, collate_fn=lambda x: x)

testloader = DataLoader(testset, batch_size=None, collate_fn=lambda x: x)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


model.encoder.register_forward_hook(get_activation("encoder"))


unreduced_bce = nn.CrossEntropyLoss()
unreduced_sparsity = nn.KLDivLoss()
unweighted_bce = nn.BCELoss()
unweighted_sparsity = nn.KLDivLoss()
optimizer = optim.AdamW(
    model.parameters(), lr=config["learning_rate"], weight_decay=0.00001
)





def save_embeddings_and_reconstructions(
    epoch, embeddings, reconstructions, config, losses, guid=RUN_UUID
):
    parent_dir = os.path.join(os.getcwd(), "outputs", "data", guid)
    config_file = os.path.join(parent_dir, "config.json")
    losses_file = os.path.join(parent_dir, "losses.tsv")
    embed_dir = os.path.join(parent_dir, "embeddings")
    recons_dir = os.path.join(parent_dir, "reconstructions")
    embed_file = os.path.join(embed_dir, f"{epoch:06}.pt")
    recons_file = os.path.join(recons_dir, f"{epoch:06}.pt")
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(recons_dir, exist_ok=True)
    if not os.path.exists(config_file):
        with open(config_file, "w+") as f:
            json.dump(config, f)
    loss_strings = [str(epoch)] + [f"{loss.tolist():.10f}" for loss in losses]
    with open(losses_file, "a") as f:
        f.write("\t".join(loss_strings) + "\n")

    torch.save(embeddings, embed_file)
    torch.save(reconstructions, recons_file)


def calculate_loss(data, masks, maskinputs, model, config):
    eps = 1e-8
    optimizer.zero_grad()
    model = model
    data = data
    outputs = model(data, masks=masks, maskinputs=maskinputs)
    # unweighted_bce_loss = unweighted_bce(output_abs, data_abs) + unweighted_bce(output_norms, data_norms) + unweighted_bce(output_durs, data_durs)
    unweighted_bce_loss = unweighted_bce(outputs, data)
    unweighted_kl_loss = torch.mean(model.encoder.kl)
    weighted_kl_loss = unweighted_kl_loss * config["kld_weight"]

    # weighted_bce_loss = (torch.mean(unreduced_bce(output_abs,F.softmax(data_abs, dim=KL_SOFTMAX_DIM)) + unreduced_bce(output_norms,F.softmax(data_norms, dim=KL_SOFTMAX_DIM)))  + unweighted_bce(output_durs, data_durs)) * config['bce_weight']
    weighted_bce_loss = torch.mean(unreduced_bce(outputs + eps, F.softmax(data, dim=KL_SOFTMAX_DIM)))* config["bce_weight"]

    return (
        weighted_bce_loss,
        weighted_kl_loss,
        unweighted_bce_loss,
        unweighted_kl_loss,
        data.detach(),
        outputs.detach(),
    )


# bach = music21.corpus.parse('bach/bwv66.6')
sample_targets, dense_targets, sample_labels, sample_masks = score_to_model_inputs("~/local_corpus/bach/chorales_006005b_(c)greentree.mid")
sample_mask_inputs = torch.unsqueeze(sample_masks, dim=-1)
sample_targets = F.one_hot(dense_targets, 129)[:, :, 1:].to(torch.float)
sample_masks = sample_mask_inputs.expand(sample_targets.shape)

sample_color_strings = [
    json.dumps(list(n), separators=(",", ":")) for n in sample_labels
]
print(sample_color_strings)

# define the training function
def fit(model, dataloader, epoch, duration_set):
    print("Training")
    running_weighted_kl_loss = 0.0
    running_weighted_bce_loss = 0.0
    running_kl_loss = 0.0
    running_bce_loss = 0.0
    counter = 0
    paths = setup_paths(RUN_UUID) 
    weighted_bces, weighted_kls, unweighted_bces, unweighted_kls = [], [], [], []
    for i, dataList in tqdm(
        enumerate(dataloader), total=TRAIN_NUM // config["batch_size"]
    ):
        model.train()
        data = dataList[0].to(device)
        metdata = dataList[1]
        masks = dataList[2]
        maskinputs = dataList[3]
        counter += 1
        (
            weighted_bce_loss,
            weighted_kl_loss,
            unweighted_bce_loss,
            unweighted_kl_loss,
            img,
            recons,
        ) = calculate_loss(data, masks, maskinputs, model, config)
        loss = weighted_bce_loss + weighted_kl_loss
        loss.backward()
        optimizer.step()
        weighted_bce_loss_item = weighted_bce_loss.item()
        weighted_kl_loss_item = weighted_kl_loss.item()
        unweighted_bce_loss_item = unweighted_bce_loss.item()
        unweighted_kl_loss_item = unweighted_kl_loss.item()
        running_weighted_bce_loss += weighted_bce_loss_item
        running_weighted_kl_loss += weighted_kl_loss_item
        running_bce_loss += unweighted_bce_loss_item
        running_kl_loss += unweighted_kl_loss_item
        weighted_bces.append(weighted_bce_loss_item)
        weighted_kls.append(weighted_kl_loss_item)
        unweighted_bces.append(unweighted_bce_loss_item)
        unweighted_kls.append(unweighted_kl_loss_item)
        epoch_weighted_bce_loss = running_weighted_bce_loss / counter
        epoch_weighted_kl_loss = running_weighted_kl_loss / counter
        epoch_weighted_loss = epoch_weighted_bce_loss + epoch_weighted_kl_loss
        epoch_bce_loss = running_bce_loss / counter
        epoch_kl_loss = running_kl_loss / counter
        epoch_loss = epoch_bce_loss + epoch_kl_loss

        if i % config['create_image_every'] == 0:
            tqdm.write(f"{RUN_UUID} epoch loss: {epoch_weighted_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch bce loss: {epoch_weighted_bce_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch kl loss: {epoch_weighted_kl_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch unweighted loss: {epoch_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch unweighted bce loss: {epoch_bce_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch unweighted kl loss: {epoch_kl_loss:.10f}")
            img_filename = os.path.join(paths['image_dir'], f"{epoch:06}-{i:06}.png")
            with torch.no_grad():
                all_embeddings = activation["encoder"]
                model.eval()
                sample_outputs = model(sample_targets.to(device), masks=sample_masks, maskinputs=sample_mask_inputs)
                sample_embeddings = activation["encoder"]
                color_strings = [met[0] for met in metdata]
                color_strings_distinct = {
                    color_string for color_string in color_strings
                }
                color_strings_distinct.update(sample_color_strings)
                color_map = {v: k for k, v in enumerate(color_strings_distinct)}
                colors = [color_map[color_string] for color_string in color_strings]
                sample_colors = [
                    color_map[color_string] for color_string in sample_color_strings
                ]

                unweighted_losses = torch.stack(
                    [
                        torch.tensor(unweighted_bces),
                        torch.tensor(unweighted_kls),
                    ],
                    0,
                )
                weighted_losses = torch.stack(
                    [
                        torch.tensor(weighted_bces),
                        torch.tensor(weighted_kls),
                    ],
                    0,
                )
                plot_details(
                    img,
                    recons,
                    all_embeddings,
                    colors,
                    epoch,
                    0,
                    filename=img_filename,
                    unweighted_losses=unweighted_losses,
                    weighted_losses=weighted_losses,
                    sample_targets=sample_targets,
                    sample_outputs=sample_outputs,
                    sample_embeddings=sample_embeddings,
                    sample_colors=sample_colors,
                    **config,
                )
                weighted_bces, weighted_kls, unweighted_bces, unweighted_kls = (
                    [],
                    [],
                    [],
                    [],
                )

    return (
        epoch_weighted_bce_loss,
        epoch_weighted_kl_loss,
        epoch_weighted_loss,
        epoch_bce_loss,
        epoch_kl_loss,
        epoch_loss,
        metdata,
    )


def validate(model, dataloader, epoch):
    model.eval()
    running_weighted_kl_loss = 0.0
    running_weighted_bce_loss = 0.0

    running_kl_loss = 0.0
    running_bce_loss = 0.0

    counter = 0
    with torch.no_grad():
        for dataList in tqdm(dataloader, total=TRAIN_NUM // config["batch_size"]):
            data = dataList[0].to(device)
            metdata = dataList[1]
            masks = dataList[2]
            maskinputs = dataList[3]
            counter += 1

            (
                weighted_bce_loss,
                weighted_kl_loss,
                unweighted_bce_loss,
                unweighted_kl_loss,
                img,
                outputs,
            ) = calculate_loss(data, masks, maskinputs, model, config)
            running_weighted_bce_loss += weighted_bce_loss.item()
            running_weighted_kl_loss += weighted_kl_loss.item()
            running_bce_loss += unweighted_bce_loss.item()
            running_kl_loss += unweighted_kl_loss.item()

            epoch_weighted_bce_loss = running_weighted_bce_loss / counter
            epoch_weighted_kl_loss = running_weighted_kl_loss / counter
            epoch_weighted_loss = epoch_weighted_bce_loss + epoch_weighted_kl_loss
            epoch_bce_loss = running_bce_loss / counter
            epoch_kl_loss = running_kl_loss / counter
            epoch_loss = epoch_bce_loss + epoch_kl_loss

            tqdm.write(f"{RUN_UUID} epoch loss: {epoch_weighted_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch bce loss: {epoch_weighted_bce_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch kl loss: {epoch_weighted_kl_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch unweighted loss: {epoch_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch unweighted bce loss: {epoch_bce_loss:.10f}")
            tqdm.write(f"{RUN_UUID} epoch unweighted kl loss: {epoch_kl_loss:.10f}")
    return (
        epoch_weighted_bce_loss,
        epoch_weighted_kl_loss,
        epoch_weighted_loss,
        epoch_bce_loss,
        epoch_kl_loss,
        epoch_loss,
        img,
        outputs,
        metdata,
    )


train_loss = []
val_bce_loss = []
val_kl_loss = []
val_sparsity_loss = []
val_loss = []
val_weighted_bce_loss = []
val_weighted_kl_loss = []
val_weighted_sparsity_loss = []

val_weighted_loss = []
start = time.time()


files = glob.glob("./outputs/images/*")
for f in files:
    os.remove(f)

dur_set = set()

for epoch in range(START_EPOCH, EPOCHS):
    print(f"Epoch {epoch} of {EPOCHS}")
    train_epoch_loss = fit(model, trainloader, epoch, dur_set)
    (
        epoch_weighted_bce_loss,
        epoch_weighted_kl_loss,
        epoch_weighted_loss,
        epoch_bce_loss,
        epoch_kl_loss,
        epoch_loss,
        img,
        outputs,
        metdata,
    ) = validate(model, testloader, epoch)
    train_loss.append(train_epoch_loss)
    val_bce_loss.append(epoch_bce_loss)
    val_kl_loss.append(epoch_kl_loss)
    val_loss.append(epoch_loss)
    val_weighted_bce_loss.append(epoch_weighted_bce_loss)
    val_weighted_kl_loss.append(epoch_weighted_kl_loss)

    val_weighted_loss.append(epoch_weighted_loss)
    paths = setup_paths(RUN_UUID)
 
    img_filename = os.path.join(paths['image_dir'], f"{epoch:06}.png")

    idx = epoch - START_EPOCH

    if idx % 1 == 0:
        with torch.no_grad():
            all_embeddings = activation["encoder"]
            save_model_state(epoch, config, model, RUN_UUID)
            sample_outputs = model(sample_targets.to(device), masks=sample_masks, maskinputs=sample_mask_inputs)
            sample_embeddings = activation["encoder"]
            losses = [
                epoch_weighted_bce_loss,
                epoch_weighted_kl_loss,
                epoch_weighted_loss,
                epoch_bce_loss,
                epoch_kl_loss,
                epoch_loss,
            ]
            unweighted_losses = torch.stack(
                [
                    torch.tensor(val_bce_loss),
                    torch.tensor(val_kl_loss),
                ],
                0,
            )
            weighted_losses = torch.stack(
                [
                    torch.tensor(val_weighted_bce_loss),
                    torch.tensor(val_weighted_kl_loss),
                ],
                0,
            )
            color_strings = [met[0] for met in metdata]
            color_strings_distinct = {
                color_string for color_string in color_strings
            }
            color_strings_distinct.update(sample_color_strings)
            color_map = {v: k for k, v in enumerate(color_strings_distinct)}
            colors = [color_map[color_string] for color_string in color_strings]
            sample_colors = [
                color_map[color_string] for color_string in sample_color_strings
            ]
            plot_details(
                img,
                outputs,
                all_embeddings,
                colors,
                epoch,
                idx,
                filename=img_filename,
                unweighted_losses=unweighted_losses,
                weighted_losses=weighted_losses,
                sample_targets=sample_targets,
                sample_outputs=sample_outputs,
                sample_embeddings=sample_embeddings,
                sample_colors=sample_colors,
                **config,
            )

end = time.time()
save_model_state(epoch, model)
print(f"{(end-start)/ORIGINAL_DIM:.3} minutes")
