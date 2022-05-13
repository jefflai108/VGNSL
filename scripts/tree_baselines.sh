#!/bin/bash 

# LJspeech tree baslines 
python data/trivial_baselines.py --gt_tree_pth data/LJspeech/Freda-formatting/dev_ground-truth-LJdev.txt
python data/trivial_baselines.py --gt_tree_pth data/LJspeech/Freda-formatting/eval1_ground-truth-LJeval1.txt

# SpokenCOCO tree baselines 
python data/trivial_baselines.py --gt_tree_pth data/SpokenCOCO/Freda-formatting/test_ground-truth-83k-5k-5k.txt
python data/trivial_baselines.py --gt_tree_pth data/SpokenCOCO/Freda-formatting/test_phn-level-ground-truth-83k-5k-5k.txt
