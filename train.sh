#!/bin/bash

set -x
# echo "args: $@"s
DATADIR=/nfs/dust/cms/user/sewuchte/GNN_weaver/LeptonID/inputData/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/PNETLepton_Muon_v0_RunIISummer20UL18MiniAODv2-106X_v16-v2/240209_164138/
model=ParT
archopts="$@"
echo $archopts
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/ParT.py"
    batchopts=" --batch-size 128"
elif [[ "$model" == "PNXT" ]]; then
    modelopts="networks/PNXT.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/PN.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "MLP" ]]; then
    modelopts="networks/MLP.py"
    batchopts="--batch-size 4096 --start-lr 1e-1"
else
    echo "Invalid model $model!"
    exit 1
fi

# set a comment via `COMMENT`
suffix=${COMMENT}

python ../weaver-core/weaver/train.py --weaver-mode class --data-train \
    "part0:${DATADIR}/*/*.root" \
    --samples-per-epoch 10000000 --train-val-split 0.8 \
    --data-config data/Muon_${model}.yaml --network-config $modelopts \
    --model-prefix training_Muon/${model}/net \
    --num-workers 2 --fetch-by-files --fetch-step 1 $batchopts \
    --num-epochs 20 --gpus 0 \
    --optimizer ranger --log logs_Muon/Muon_${model}.log \
    --tensorboard Muon_${model}  $archopts --start-lr 1e-4 --cross-validation 'luminosityBlock%5'
