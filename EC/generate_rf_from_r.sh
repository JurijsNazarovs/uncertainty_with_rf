#!/bin/bash



rfType=${1:-"gauss"} #chisq, gauss
df=${2:-1}
alpha=${3:-0.05}
n=${4:-10000}
dimX=${5:-32}
dimY=${6:-32}
scale=${7:-10}  #10 = 32,64, 100-304 #hyper parameter
prop=1
#prop=${8:-1} # proportion of significant fields save; 1: within, 0: between


#outPath="./generated_rf/depth/304/generated_rf_${prop}_adj/"
outPath="./generated_rf/${dimX}_${n}/generated_rf_${prop}_scale_${scale}/"

if [[ $rfType = "gauss" ]]; then
  outPath="$outPath/$rfType"
else
  outPath="$outPath/$rfType/df$df"
fi
thresh=$(python3 rf_get_threshold.py --test rf --rf_type $rfType --df $df\
                 --scale $scale --L $dimX $dimY --alpha $alpha)
echo "thresh=$thresh"
outPathMask="${outPath}_mask"

#mkdir -p $(dirname "$outPath")
#mkdir -p $(dirname "$outPathMask")
mkdir -p "$outPath"
mkdir -p "$outPathMask"
Rscript generate_rf.R "$outPath/" "$n"  "$dimX" "$dimY" "$rfType" "$covType" \
        "$df" "$scale" "$thresh" "$outPathMask/" "$prop"


# python3 main.py --n_epochs 300 --data mnist --experimentID mnist_within_test_meanvar_gen01 --df 5  --method within --device 0 --lr 0.0001 --batch_size 100 --normalize_gan --gen_loss_weight 0.1
