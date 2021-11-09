#!/usr/bin/env bash
python search.py --phase train --dataroot database/cityscapes \
  --restore_G_path logs/pix2pix/cityscapes/supernet/checkpoints/latest_net_G.pth \
  --output_path logs/pix2pix/cityscapes/supernet/results.pkl \
  --batch_size 4 --config_set channels-48 \
  --real_stat_path real_stat/cityscapes_A.npz \
  --drn_path drn-d-105_ms_cityscapes.pth \
  --cityscapes_path database/cityscapes-origin \
  --table_path datasets/train_table.txt \
  --direction BtoA --no_fid \
  --meta_path datasets/metas/cityscapes/train2.meta
