exp="default"
gpu_num="1"

model="r50_deaotl"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_aotl"
# model="swinb_aotl"

# Training ##
stage="dav"
python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num}

# Evaluation ##
dataset="davis2017"
split="test"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}
    
dataset="davis2017"
split="val"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num}