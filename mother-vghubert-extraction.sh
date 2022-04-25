#!/bin/bash 

stage=$1

if [ $stage -eq 0 ]; then # done
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh val disc-81 $layer
done 
fi 

if [ $stage -eq 1 ]; then # done
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh val disc-82 $layer
done 
fi 

if [ $stage -eq 2 ]; then # done
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh val hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 3 ]; then # done
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh val hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 4 ]; then # done 
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh test disc-81 $layer
done 
fi 

if [ $stage -eq 5 ]; then # done 
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh test disc-82 $layer
done 
fi 

if [ $stage -eq 6 ]; then # done
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh test hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 7 ]; then # done
for layer in 0 1 2 3 4 5 6 7 8 9 10 11; do
    ./scripts/data_reformat-vghubert.sh test hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 8 ]; then 
for layer in 1 2; do
    ./scripts/data_reformat-vghubert.sh train disc-81 $layer
done 
fi 

if [ $stage -eq 9 ]; then 
for layer in 4 5; do
    ./scripts/data_reformat-vghubert.sh train disc-81 $layer
done 
fi 

if [ $stage -eq 10 ]; then 
for layer in 7 8; do
    ./scripts/data_reformat-vghubert.sh train disc-81 $layer
done 
fi 

if [ $stage -eq 11 ]; then 
for layer in 10 11; do
    ./scripts/data_reformat-vghubert.sh train disc-81 $layer
done 
fi 

if [ $stage -eq 12 ]; then 
for layer in 1 2; do
    ./scripts/data_reformat-vghubert.sh train disc-82 $layer
done 
fi 

if [ $stage -eq 13 ]; then 
for layer in 4 5; do
    ./scripts/data_reformat-vghubert.sh train disc-82 $layer
done 
fi 

if [ $stage -eq 14 ]; then 
for layer in 7 8; do
    ./scripts/data_reformat-vghubert.sh train disc-82 $layer
done 
fi 

if [ $stage -eq 15 ]; then 
for layer in 10 11; do
    ./scripts/data_reformat-vghubert.sh train disc-82 $layer
done 
fi 

if [ $stage -eq 16 ]; then 
for layer in 0; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 17 ]; then 
for layer in 1; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 18 ]; then 
for layer in 2; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 19 ]; then 
for layer in 3; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 20 ]; then 
for layer in 4; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 21 ]; then 
for layer in 5; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 22 ]; then 
for layer in 6; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 23 ]; then 
for layer in 7; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 24 ]; then 
for layer in 8; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 25 ]; then 
for layer in 9; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 26 ]; then 
for layer in 10; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 27 ]; then 
for layer in 11; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-81 $layer
done 
fi 

if [ $stage -eq 28 ]; then 
for layer in 0; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 29 ]; then 
for layer in 1; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 30 ]; then 
for layer in 2; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 31 ]; then 
for layer in 3; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 32 ]; then 
for layer in 4; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 33 ]; then 
for layer in 5; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-82 $layer
done 
fi 

if [ $stage -eq 34 ]; then 
for layer in 6; do
    ./scripts/data_reformat-vghubert.sh train hubert2_cat_disc-82 $layer
done 
fi 


exit 0





