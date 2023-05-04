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


from utils import ChoraleChords, plot_embedding_space, save_interleaved, plot_details


from VAEModels import GaussianVAE

device = "cpu" # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
# 13k 430
ORIGINAL_DIM = 128
LATENT_DIM = 2
MIN_MIDI = 30
MAX_MIDI = 90
DATA_LENGTH = MAX_MIDI - MIN_MIDI
INTERMEDIATE_DIMS = ORIGINAL_DIM*32
LEARNING_RATE = 1e-2                                                                                                                            
RHO  = 0.0074
BETA = 4
KLD_WEIGHT = 0.0 #
WEIGHT_MULT = 1.# 1e3
SPARSE_WEIGHT = 1
RUN_UUID = str(uuid.uuid1())
BATCH_SIZE=256
KL_SOFTMAX_DIM=1
EPOCHS=5000
START_EPOCH=0

default_config = {
    'original_dim': ORIGINAL_DIM,
    'intermediate_dim': INTERMEDIATE_DIMS,
    'latent_dim': LATENT_DIM,
    'data_length': DATA_LENGTH,
    'learning_rate': LEARNING_RATE,
    'kld_weight': KLD_WEIGHT,
    'bce_weight': WEIGHT_MULT,
    'sparsity_weight': SPARSE_WEIGHT,
    'batch_size': BATCH_SIZE,
    'class_weight_exponent': 0.25,
    'weight_inverse_frequency' : False,
    'kl_softmax_dim': 0
}

resume_run = None # "07750700-e6c4-11ed-a95b-3af9d377b57f"

if resume_run is not None:
    RUN_UUID=resume_run
    parent_dir = os.path.join(os.getcwd() , "outputs", "data", resume_run)
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
    with open("./config.json", 'r') as f:
        config = json.load(f)
    model = GaussianVAE(**config).to(device)


summary(model,(1, DATA_LENGTH))
trainset = ChoraleChords(augmented=True)
testset = ChoraleChords(augmented=True)
trainset.encoding = 'multi_hot'
testset.encoding = 'multi_hot'
# trainloader
trainloader = DataLoader(
    trainset, 
    batch_size=config['batch_size'],
    shuffle=True
)
#testloader
testloader = DataLoader(
    testset, 
    batch_size=len(testset), 
    shuffle=False
)
trainset.encoding = 'multi_hot'
testset.encoding = 'multi_hot'
#model.apply(weight_init)
print(testset[0:5].shape)
model.eval()
with torch.no_grad():
    outputs = model(testset[0:5])

activation = {}

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
model.encoder.register_forward_hook(get_activation('encoder'))

print(f"testset.uniques.shape: {torch.squeeze(testset[:]).shape}")
colors = torch.mean(testset.transposed_dense.to(torch.float),1)

test_img = testset.original_mh
batch_len = test_img.size(0)
test_img = test_img.view([batch_len, DATA_LENGTH])
test_img.to(device)
model.eval()
with torch.no_grad():
    outputs = model(test_img)
#plot_embedding_space(activation['encoder'], colors)
#weight=(1 - torch.mean(torch.squeeze(testset[:]), dim=0))
#weight=(1 / torch.sum(torch.squeeze(testset[:]), dim=0))
# weight=(1 / torch.mean(torch.squeeze(testset[:]), dim=0))
#weights = (1 - torch.mean(torch.squeeze(testset[:]), dim=0))
#weights = (1 / 0.2 * torch.mean(torch.squeeze(testset[:]), dim=0))
#weights = torch.pow((1 / torch.mean(torch.squeeze(testset[:]), dim=0)), config['class_weight_exponent'])
#weight=weights

if config['weight_inverse_frequency']:
    weights = torch.pow((1 / torch.mean(torch.squeeze(testset[:]), dim=0)), config['class_weight_exponent'])
else:
    weights = torch.ones([DATA_LENGTH])

weights = (1 - torch.mean(torch.squeeze(testset[:]), dim=0))
num_pos_by_pitch = torch.sum(testset[:], dim=0)
num_neg_by_pitch = len(testset[:]) - num_pos_by_pitch
pos_weight = num_neg_by_pitch / num_pos_by_pitch

#unreduced_bce = nn.CrossEntropyLoss()
unreduced_bce = nn.CrossEntropyLoss()
unreduced_sparsity= nn.KLDivLoss()
unweighted_bce = nn.BCELoss()
unweighted_sparsity = nn.KLDivLoss()
optimizer = optim.Adagrad(model.parameters(), lr=config['learning_rate'])


SAMPLE_INDICES = torch.randint(len(trainset),(DATA_LENGTH,))

def save_model_state(epoch, model, guid = RUN_UUID):
    parent_dir = os.path.join(os.getcwd() , "outputs", "data", guid)
    model_dir = os.path.join(parent_dir, "model_state")
    model_file = os.path.join(model_dir, f"{epoch}.pth")
    config_file = os.path.join(parent_dir, "config.json")
    if not os.path.exists(config_file):
        with open(config_file, "w+") as f:
            json.dump(config, f)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_file)

def save_embeddings_and_reconstructions(epoch, embeddings, reconstructions, config, losses, guid = RUN_UUID):
    parent_dir = os.path.join(os.getcwd() , "outputs", "data", guid)
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
    with open(losses_file, 'a') as f:
        f.write('\t'.join(loss_strings) + "\n")
    
    torch.save(embeddings, embed_file)
    torch.save(reconstructions, recons_file)

def kl_divergence(rho, rho_hat):
    return (rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

def calculate_loss(data, model, config):
    eps = 1e-6
    img =  data
    img = img.to(device)
    optimizer.zero_grad()
    outputs = model(img)

    unweighted_bce_loss = unweighted_bce(outputs, img)    
    unweighted_kl_loss =  model.encoder.kl.mean()
    weighted_kl_loss = unweighted_kl_loss * config['kld_weight']



    #handrolled_sparsity = kl_divergence(F.softmax(outputs, dim=KL_SOFTMAX_DIM), F.softmax(img, dim=KL_SOFTMAX_DIM))
    #print(f"kl div:{handrolled_sparsity}")
    if config['weight_inverse_frequency']:
        chordwise_freq = weights * img
        chordwise_weights = torch.mean(chordwise_freq, dim=1)
        unreduced_bce_loss = unreduced_bce(outputs,F.softmax(img, dim=KL_SOFTMAX_DIM))
        #print(f"unreduced bce loss: {unreduced_bce_loss}, {unreduced_bce_loss.shape}")
        batch_mean_bce = unreduced_bce_loss #torch.mean(unreduced_bce_loss, dim=1)
        weighted_bce_loss = torch.mean(chordwise_weights * batch_mean_bce, dim=0) * config['bce_weight']
    else: 
        weighted_bce_loss = unreduced_bce(outputs,F.softmax(img, dim=KL_SOFTMAX_DIM)) * config['bce_weight']

    return weighted_bce_loss, weighted_kl_loss, unweighted_bce_loss, unweighted_kl_loss, img.detach(), outputs.detach()


# define the training function
def fit(model, dataloader, epoch):
    print('Training')
    model.train()
    running_weighted_kl_loss = 0.0
    running_weighted_bce_loss = 0.0
    running_kl_loss = 0.0
    running_bce_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(trainset)/dataloader.batch_size)):
        counter += 1
        weighted_bce_loss, weighted_kl_loss, unweighted_bce_loss, unweighted_kl_loss,_, _ = calculate_loss(data, model, config)
        loss = weighted_bce_loss + weighted_kl_loss
        loss.backward()
        optimizer.step()
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
    print(f"{RUN_UUID} epoch loss: {epoch_weighted_loss:.10f}")
    print(f"{RUN_UUID} epoch bce loss: {epoch_weighted_bce_loss:.10f}")
    print(f"{RUN_UUID} epoch kl loss: {epoch_weighted_kl_loss:.10f}")
    print(f"{RUN_UUID} epoch unweighted loss: {epoch_loss:.10f}")
    print(f"{RUN_UUID} epoch unweighted bce loss: {epoch_bce_loss:.10f}")
    print(f"{RUN_UUID} epoch unweighted kl loss: {epoch_kl_loss:.10f}")

    return (epoch_weighted_bce_loss, epoch_weighted_kl_loss, epoch_weighted_loss, epoch_bce_loss, epoch_kl_loss, epoch_loss)


def validate(model, dataloader, epoch):
    parent_dir = os.path.join(os.getcwd() , "outputs", "data", RUN_UUID)
    config_file = os.path.join(parent_dir, "config.json")
    losses_file = os.path.join(parent_dir, "losses.tsv")
    embed_dir = os.path.join(parent_dir, "embeddings")
    recons_dir = os.path.join(parent_dir, "reconstructions")
    image_dir = os.path.join(parent_dir, "images")
    embed_file = os.path.join(embed_dir, f"{epoch:06}.pt")
    recons_file = os.path.join(recons_dir, f"{epoch:06}.pt")
    img_filename = os.path.join(image_dir, f"{epoch:06}.png")
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(recons_dir, exist_ok=True)
    model.eval()
    running_weighted_kl_loss = 0.0
    running_weighted_bce_loss = 0.0

    running_kl_loss = 0.0
    running_bce_loss = 0.0

    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(testset)/dataloader.batch_size)):
            counter += 1
            
            weighted_bce_loss, weighted_kl_loss, unweighted_bce_loss, unweighted_kl_loss,img, outputs = calculate_loss(data, model, config)
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
        # save the reconstructed images 
        if epoch % 100 == 0:
            save_model_state(epoch, model)
    print(f"{RUN_UUID} epoch loss: {epoch_weighted_loss:.10f}")
    print(f"{RUN_UUID} epoch bce loss: {epoch_weighted_bce_loss:.10f}")
    print(f"{RUN_UUID} epoch kl loss: {epoch_weighted_kl_loss:.10f}")
    print(f"{RUN_UUID} epoch unweighted loss: {epoch_loss:.10f}")
    print(f"{RUN_UUID} epoch unweighted bce loss: {epoch_bce_loss:.10f}")
    print(f"{RUN_UUID} epoch unweighted kl loss: {epoch_kl_loss:.10f}")
    return (epoch_weighted_bce_loss, epoch_weighted_kl_loss, epoch_weighted_loss, epoch_bce_loss, epoch_kl_loss, epoch_loss,img, outputs)


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


files = glob.glob('./outputs/images/*')
for f in files:
    os.remove(f)
        
bach = music21.corpus.parse('bach/bwv66.6')
sample_targets = torch.tensor([])
sample_labels = torch.tensor([])


for chord in bach.chordify().recurse().getElementsByClass(['Chord', 'Measure']):
        if 'Chord' in chord.classes:
            chord_zeroes = torch.zeros(128,dtype=torch.float)
            midis = np.array([p.midi for p in chord.pitches])
            chord_zeroes[midis] = 1.
            midis.resize(10)
            dense = torch.tensor(midis, dtype=torch.float)
            sample_targets = torch.cat([sample_targets, torch.unsqueeze(chord_zeroes[MIN_MIDI:MAX_MIDI], dim=0)], dim=0)
            sample_labels = torch.cat([sample_labels, torch.unsqueeze(dense, dim=0)], dim=0)


for epoch in range(START_EPOCH, EPOCHS):
    print(f"Epoch {epoch} of {EPOCHS}")
    train_epoch_loss = fit(model, trainloader, epoch)
    epoch_weighted_bce_loss, epoch_weighted_kl_loss, epoch_weighted_loss, epoch_bce_loss, epoch_kl_loss, epoch_loss, img, outputs = validate(model, testloader, epoch)
    train_loss.append(train_epoch_loss)
    val_bce_loss.append(epoch_bce_loss)
    val_kl_loss.append(epoch_kl_loss)
    val_loss.append(epoch_loss)
    val_weighted_bce_loss.append(epoch_weighted_bce_loss)
    val_weighted_kl_loss.append(epoch_weighted_kl_loss)

    val_weighted_loss.append(epoch_weighted_loss)
    parent_dir = os.path.join(os.getcwd() , "outputs", "data", RUN_UUID)
    config_file = os.path.join(parent_dir, "config.json")
    losses_file = os.path.join(parent_dir, "losses.tsv")
    embed_dir = os.path.join(parent_dir, "embeddings")
    recons_dir = os.path.join(parent_dir, "reconstructions")
    image_dir = os.path.join(parent_dir, "images")
    embed_file = os.path.join(embed_dir, f"{epoch:06}.pt")
    recons_file = os.path.join(recons_dir, f"{epoch:06}.pt")
    img_filename = os.path.join(image_dir, f"{epoch:06}.png")
    os.makedirs(embed_dir, exist_ok=True)
    os.makedirs(recons_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    idx = epoch - START_EPOCH

    if idx % 2 == 0:
        all_embeddings = activation['encoder']
        with torch.no_grad():
            sample_outputs = model(sample_targets)
            sample_embeddings = activation['encoder']
            losses = [epoch_weighted_bce_loss, epoch_weighted_kl_loss, epoch_weighted_loss, epoch_bce_loss, epoch_kl_loss, epoch_loss]
            unweighted_losses = torch.stack(
                [ 
                    torch.tensor(val_bce_loss),
                    torch.tensor(val_kl_loss), 
                ],
                0
            )
            weighted_losses = torch.stack(
                [                     
                    torch.tensor(val_weighted_bce_loss),
                    torch.tensor(val_weighted_kl_loss), 
                ],
                0
            )

            #save_embeddings_and_reconstructions(epoch, activation['encoder'], outputs, config, losses)
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
                **config
            )

end = time.time()
save_model_state(epoch, model)
print(f"{(end-start)/ORIGINAL_DIM:.3} minutes")
# save the trained model
#torch.save(model.state_dict(), f"../outputs/sparse_ae{EPOCHS}.pth")