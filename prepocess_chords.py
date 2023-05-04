#!/usr/bin/env python3

import music21
import numpy as np
import torch
from tqdm import tqdm
import glob
import os

all_intervals = set()
all_midis = set()
all_mh = torch.tensor([], dtype=torch.float)
all_dense = torch.tensor([],dtype=torch.float)
for idx, score in tqdm(enumerate(music21.corpus.corpora.CoreCorpus().search('bach', 'composer'))):
    work = score.parse()
    for chord in work.chordify().recurse().getElementsByClass(['Chord']):
        all_midis.add(tuple([p.midi for p in chord.pitches]))

set_size = len(all_midis)
mh_zeros = torch.zeros([set_size, 128])
mh_ones = torch.ones([set_size,128])
nested_midis = torch.nested.nested_tensor(list(all_midis), dtype=torch.int64, requires_grad=False, pin_memory=True)
all_dense = torch.nested.to_padded_tensor(nested_midis, 0.0, output_size=[set_size,10]).to(torch.int64)
all_mh = mh_zeros.scatter_(1, all_dense, mh_ones)

print(all_mh.shape)
print(all_dense.shape)
torch.save(all_mh,  f'./original_multihot.pt')
torch.save(all_dense, f'./original_dense.pt')
exit()
'''
all_intervals = set()
all_midis = set()
all_mh = torch.tensor([], dtype=torch.float)
all_dense = torch.tensor([],dtype=torch.float)
for idx, score in tqdm(enumerate(music21.corpus.corpora.CoreCorpus().search('bach', 'composer'))):
    work = score.parse()
    for chord in work.chordify().recurse().getElementsByClass(['Chord']):
        midis = [p.midi for p in chord.pitches]
        all_intervals.add(tuple([p - midis[0] for p in midis]))'''


print(f"{len(all_midis)} unique midi sets")

'''for c in tqdm(list(all_intervals)):
    ms = list(c)
    high = ms[-1]
    tpstion = 0
    while tpstion + high < 128:
        mst = [m + tpstion for m in ms]
        all_midis.add(tuple(mst))
        tpstion += 1
'''
mh_midis = []
dense_midis = []
print(f"dataset size is now: {len(all_midis)}")
chord_zeroes = torch.zeros(128, dtype=torch.float)
for c in tqdm(list(all_midis)):
    chord_zeroes[:] = 0.0
    midis = np.array(list(c))
    chord_zeroes[midis] = 1.
    nt = torch.nested.nested_tensor([[1,2,3], [1,2,3,4,5]], dtype=torch.float, requires_grad=False, pin_memory=True)
    ntp = torch.nested.to_padded_tensor(nt, 0.0, output_size=[2,10])
    midis.resize(10)



dense = torch.tensor(midis, dtype=torch.float)
all_mh = torch.cat([all_mh, torch.unsqueeze(chord_zeroes, dim=0)], dim=0)               
all_dense = torch.cat([all_dense, torch.unsqueeze(dense, dim=0)], dim=0)

#with open('tensor_midi_test.pt', 'w+') as f:
print(all_mh.shape)
print(all_dense.shape)
torch.save(all_mh,  f'./all_multihot.pt')
torch.save(all_dense, f'./all_dense.pt')


