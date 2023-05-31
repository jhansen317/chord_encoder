import torch
from torch import nn
import json
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity




LATENT_DIM = 3
MIN_MIDI = 0
MAX_MIDI = 128
MIDI_WIDTH = 128
DURATION_WIDTH = 11  # 14359
NORMALS_WIDTH = 11
DATA_LENGTH = MAX_MIDI - MIN_MIDI #+ NORMALS_WIDTH


def kl_divergence(rho, rho_hat):
    return torch.sum(
        rho * torch.log(rho / rho_hat)
        + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )


class Decoder(nn.Module):
    def __init__(self, **config):
        super(Decoder, self).__init__()
        hidden_activation_name = config.get("hidden_activation", "tanh")
        self.device = config.get("device", "mps")
        if hidden_activation_name == "tanh":
            hidden_activation = nn.Tanh
        elif hidden_activation_name == "leaky_relu":
            hidden_activation = nn.LeakyReLU
        elif hidden_activation_name == "relu":
            hidden_activation = nn.ReLU
        self.midi_width = config.get("midi_width", MIDI_WIDTH)
        self.normal_width = config.get("normal_width", NORMALS_WIDTH)
        self.duration_width = config.get("duration_width", DURATION_WIDTH)
        self.include_normals = config.get("include_normals", False)
        self.include_durations = config.get("include_durations", False)
        self.output_width = self.midi_width

        if self.include_durations:
            self.output_width += self.duration_width

        self.midi_out_width = config["midi_out_width"]
        self.normal_out_width = config["normal_out_width"]
        self.duration_out_width = config["duration_out_width"]
        self.merge_strategy = config.get("merge_strategy", "add")

        default_dims = [128]
        hidden_dims = sorted(config.get("hidden_dims", default_dims))

        modules = []
        self.decoder_input = nn.Sequential(
            nn.Linear(config["latent_dim"] + 25, hidden_dims[0]), hidden_activation()
        )
        if self.merge_strategy != "add":
            hidden_dims[-1] = (
                self.midi_out_width + self.normal_out_width + self.duration_out_width
            )
            self.decoder_output = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.output_width), nn.Sigmoid()
            )
        else:
            self.decoder_output = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.output_width), nn.Sigmoid()
            )
        modules.append(self.decoder_input)
        for i in range(len(hidden_dims) - 1):
            if i == len(hidden_dims) - 2:
                            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    hidden_activation()
                )
            )
            else:
                modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    hidden_activation(),
                    nn.Dropout1d(p=config.get("dropout", 0.1)),
                )
            )
        modules.append(self.decoder_output)
        self.decoder = nn.Sequential(*modules)

    def forward(self, z, **kwargs):
        y = kwargs.get('maskinput', torch.zeros(len(z), 25)).to(self.device)
        z = self.decoder(torch.cat([z, y], dim=1))
        z = z.view(-1, 25, 128)
        masks = kwargs.get('masks', torch.zeros_like(z)).to(self.device)
        return torch.masked_fill(z, masks.to(bool), 0.0)
         

class DecoderConv(nn.Module):
    def __init__(self, **config):
        super(Decoder, self).__init__()
        hidden_activation_name = config.get("hidden_activation", "tanh")
        if hidden_activation_name == "tanh":
            hidden_activation = nn.Tanh
        elif hidden_activation_name == "leaky_relu":
            hidden_activation = nn.LeakyReLU
        elif hidden_activation_name == "relu":
            hidden_activation = nn.ReLU
        self.midi_width = config.get("midi_width", MIDI_WIDTH)
        self.normal_width = config.get("normal_width", NORMALS_WIDTH)
        self.duration_width = config.get("duration_width", DURATION_WIDTH)
        self.include_normals = config.get("include_normals", False)
        self.include_durations = config.get("include_durations", False)
        self.output_width = self.midi_width
        #if self.include_normals:
        #    self.output_width += self.normal_width
        if self.include_durations:
            self.output_width += self.duration_width

        self.midi_out_width = config["midi_out_width"]
        self.normal_out_width = config["normal_out_width"]
        self.duration_out_width = config["duration_out_width"]
        self.merge_strategy = config.get("merge_strategy", "add")

        default_dims = [128]
        hidden_dims = sorted(config.get("hidden_dims", default_dims))

        modules = []
        self.decoder_input = nn.Sequential(
            nn.Linear(config["latent_dim"], hidden_dims[0]), hidden_activation()
        )
        if self.merge_strategy != "add":
            hidden_dims[-1] = (
                self.midi_out_width + self.normal_out_width + self.duration_out_width
            )
            self.decoder_output = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.output_width), nn.Sigmoid()
            )
        else:
            self.decoder_output = nn.Sequential(
                nn.Linear(hidden_dims[-1], self.output_width), nn.Sigmoid()
            )
        modules.append(self.decoder_input)
        for i in range(len(hidden_dims) - 1):
            if i == len(hidden_dims) - 2:
                            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    hidden_activation()
                )
            )
            else:
                modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    hidden_activation(),
                    nn.Dropout1d(p=config.get("dropout", 0.1)),
                )
            )
        modules.append(self.decoder_output)
        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        z = self.decoder(z)
        return z


class VariationalEncoder(nn.Module):
    def __init__(self, **config):
        super(VariationalEncoder, self).__init__()
        hidden_activation_name = config.get("hidden_activation", "tanh")
        if hidden_activation_name == "tanh":
            hidden_activation = nn.Tanh
        elif hidden_activation_name == "leaky_relu":
            hidden_activation = nn.LeakyReLU
        elif hidden_activation_name == "relu":
            hidden_activation = nn.ReLU
        default_dims = [128]
        self.midi_width = config.get("midi_width", MIDI_WIDTH)
        self.midi_out_width = config["midi_out_width"]
        self.normal_width = config.get("normal_width", NORMALS_WIDTH)
        self.normal_out_width = config["normal_out_width"]

        self.duration_width = config.get("duration_width", DURATION_WIDTH)
        self.duration_out_width = config["duration_out_width"]

        self.include_normals = config.get("include_normals", False)
        self.include_durations = config.get("include_durations", False)
        self.merge_strategy = config.get("merge_strategy", "add")
        hidden_dims = sorted(config.get("hidden_dims", default_dims), reverse=True)
        modules = []

        if self.merge_strategy != "add":
            hidden_dims[0] = (
                self.midi_out_width + self.normal_out_width
            )  # + self.duration_out_width

        self.device = config.get("device", "mps")
        self.normal_input = nn.Linear(self.normal_width, self.normal_out_width)
        self.absolute_input = nn.Linear(self.midi_width, self.midi_out_width)
        self.duration_input = nn.Linear(self.duration_width, self.duration_out_width)
        self.input_norm = nn.BatchNorm1d(hidden_dims[0])
        for i in range(len(hidden_dims) - 1):
            if i == len(hidden_dims) - 2:
                            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    hidden_activation()
                )
            )
            else:
                modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    hidden_activation(),
                    nn.Dropout1d(p=config.get("dropout", 0.1)),
                )
            )
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], config["latent_dim"])
        self.fc_var = nn.Linear(hidden_dims[-1], config["latent_dim"])
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device)  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.to(self.device)
        self.kl = 0

    def forward(self, x, **kwargs):
        x = torch.flatten(x, start_dim=1) #.to(self.device)
        eps = 1e-8
        midi_slice = x[:, 0 : self.midi_width]
        x = self.absolute_input(midi_slice)
        if self.include_normals:
            normal_slice = kwargs.get('maskinput', torch.zeros(len(x), 25)).to(self.device)
            if self.merge_strategy == "add":
                x = x + self.normal_input(normal_slice)
            else:
                x = torch.cat([x, self.normal_input(normal_slice)], dim=1)

        if self.include_durations:
            duration_slice = x[
                :,
                self.midi_width
                + self.normal_width : self.midi_width
                + self.normal_width
                + self.duration_width,
            ]
            if self.merge_strategy == "add":
                x = x + self.duration_input(duration_slice)
            else:
                x = torch.cat([x, self.duration_input(duration_slice)], dim=1)
        x = self.input_norm(x)
        x = F.leaky_relu(x)
        x = self.encoder(x)  # .to(self.device)
        mu = self.fc_mu(x)  # .to(self.device)
        sigma = torch.exp(self.fc_var(x))  # .to(self.device)
        z = mu + sigma * self.N.sample(mu.shape)  # .to(self.device)
        self.kl = sigma**2 + mu**2 - torch.log(sigma + eps) - 1 / 2
        return z


class GaussianVAE(nn.Module):
    def __init__(self, **config):
        super(GaussianVAE, self).__init__()
        self.config = config
        self.encoder = VariationalEncoder(**self.config)
        self.decoder = Decoder(**self.config)

    def forward(self, x, **kwargs):
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True, with_stack=True) as prof:
            z = self.encoder(x, **kwargs)
            out = self.decoder(z, **kwargs)
        print(prof.key_averages(group_by_stack_n=1).table(sort_by="self_cpu_memory_usage", row_limit=10))
        return out
