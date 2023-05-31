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

local_corpus_path = '/Users/hansen/local_corpus'
local_corpus_midi_glob = os.path.join(local_corpus_path, '*.mid')
paths = glob.glob(local_corpus_midi_glob)
print(f"found {len(paths)} midi files to parse!")
composer_map = {
    "bach":0, 
    "schubert":1, 
    "schumann":2, 
    "beethoven":3, 
    "josquin":4, 
    "monteverdi":5,
    "mozart":6,
    "brahms":7
    }
inverse_composer_map = {
    0: "bach", 
    1: "schubert", 
    2: "schumann",
    3:"beethoven",
    4:"josquin",
    5:"monteverdi",
    6:"mozart",
    7:"brahms"
    }
for idx, score in tqdm(enumerate(paths)):
    print(f"parsing: {score}")
    try:
        work = music21.converter.parse(score)
        for chord in work.chordify().recurse().getElementsByClass(['Chord']):
            all_midis.add(tuple([p.midi for p in chord.pitches]))
        print(f"all_midis length: {len(all_midis)}")
    except Exception as e:
        print(f"got exception parsing {score}: {e}")
        continue

for idx, score in tqdm(enumerate(music21.corpus.corpora.CoreCorpus().search('bach', 'composer'))):
    work = score.parse()
    for chord in work.chordify().recurse().getElementsByClass(['Chord']):
        all_midis.add(tuple([p.midi for p in chord.pitches]))

for idx, score in tqdm(enumerate(music21.corpus.corpora.CoreCorpus().search('beethoven', 'composer'))):
    work = score.parse()
    for chord in work.chordify().recurse().getElementsByClass(['Chord']):
        all_midis.add(tuple([p.midi for p in chord.pitches]))

for idx, score in tqdm(enumerate(music21.corpus.corpora.CoreCorpus().search('josquin', 'composer'))):
    work = score.parse()
    for chord in work.chordify().recurse().getElementsByClass(['Chord']):
        all_midis.add(tuple([p.midi for p in chord.pitches]))

for idx, score in tqdm(enumerate(music21.corpus.corpora.CoreCorpus().search('mozart', 'composer'))):
    work = score.parse()
    for chord in work.chordify().recurse().getElementsByClass(['Chord']):
        all_midis.add(tuple([p.midi for p in chord.pitches]))

for idx, score in tqdm(enumerate(music21.corpus.corpora.CoreCorpus().search('monteverdi', 'composer'))):
    work = score.parse()
    for chord in work.chordify().recurse().getElementsByClass(['Chord']):
        all_midis.add(tuple([p.midi for p in chord.pitches]))



set_size = len(all_midis)
chord_lengths = [len(list(c)) for c in all_midis]
max_chord_size = max(chord_lengths)
mh_zeros = torch.zeros([set_size, 129])
mh_ones = torch.ones([set_size,129])
nested_midis = torch.nested.nested_tensor(list(all_midis), dtype=torch.int64, requires_grad=False, pin_memory=True)
all_dense = torch.nested.to_padded_tensor(nested_midis, 0.0, output_size=[set_size,max_chord_size]).to(torch.int64)
all_mh = mh_zeros.scatter_(1, all_dense, mh_ones)[:, 1:]

print(all_mh.shape)
print(all_dense.shape)
torch.save(all_mh,  f'./original_multihot.pt')
torch.save(all_dense, f'./original_dense.pt')
transposed_midis = set()

for idx, chord in tqdm(enumerate(all_midis)):
    ints = [p - chord[0] for p in chord]
    high = ints[-1]
    tpstion = 1
    while tpstion + high < 129:
        mst = [m + tpstion for m in ints]
        transposed_midis.add(tuple(mst))
        tpstion += 1

set_size = len(transposed_midis)
mh_zeros = torch.zeros([set_size, 129])
mh_ones = torch.ones([set_size,129])
nested_midis = torch.nested.nested_tensor(list(transposed_midis), dtype=torch.int64, requires_grad=False, pin_memory=True)
all_dense = torch.nested.to_padded_tensor(nested_midis, 0.0, output_size=[set_size,max_chord_size]).to(torch.int64)
all_mh = mh_zeros.scatter_(1, all_dense, mh_ones)[:, 1:]


print(f"{len(transposed_midis)} unique midi sets")
print(all_mh.shape)
print(all_dense.shape)
torch.save(all_mh,  f'./all_multihot.pt')
torch.save(all_dense, f'./all_dense.pt')


'''for c in tqdm(list(all_intervals)):
    ms = list(c)
    high = ms[-1]
    tpstion = 0
    while tpstion + high < 128:
        mst = [m + tpstion for m in ms]
        all_midis.add(tuple(mst))
        tpstion += 1

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
torch.save(all_dense, f'./all_dense.pt')'''


