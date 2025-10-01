## Contains implementations of Muon & MuonAll optimizer

pip install git+https://github.com/Saurabh750/optimizer

## To use Muon optimizer:
from muon import Muon
optimizer = Muon(model.named_parameters(), lr=MUON_LEARNING_RATE, adamw_lr=ADAMW_LEARNING_RATE)

## To use MuonAll optimizer:
from muon import MuonAll
optimizer = MuonAll(model.named_parameters(), lr=MUON_LEARNING_RATE)

