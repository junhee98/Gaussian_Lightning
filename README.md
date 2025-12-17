# Compressive 3D Gaussian Splatting via Frequency Adaptive Optimization


## Setup
#### Local Setup
The codebase is based on [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)

The used datasets, MipNeRF360 and Tank & Temple, are hosted by the paper authors [here](https://jonbarron.info/mipnerf360/). 

For installation:
```
git clone --recursive https://github.com/junhee98/Gaussian_Lightning.git
cd Gaussian_Lightning
# if you have already cloned Gaussian_Lightning:
# git submodule update --init --recursive
```
```shell
conda env create --file environment.yml
conda activate gaussianlightning
```
note: we modified the "diff-gaussian-rasterization" in the submodule to get the Global Significant Score.


## Compress to Compact Representation

Our code includes **3 ways** to make the 3D Gaussians be compact
<!-- #### Option 0 Run all (currently Prune + SH distillation) -->


#### Option 1 Prune & Recovery
Users can directly prune a trained 3D-GS checkpoint using the following command (default setting):
```
bash scripts/run_prune_finetune.sh
```

Users can also train from scratch and jointly prune redundant Gaussians in training using the following command (different setting from the paper):
```
bash scripts/run_train_densify_prune.sh
```
note: 3D-GS is trained for 20,000 iterations and then prune it. The resulting ply file is approximately 35% of the size of the original 3D-GS while ensuring a comparable quality level.


#### Option 2 SH distillation
Users can distill 3D-GS checkpoint using the following command (default setting):
```
bash scripts/run_distill_finetune.sh
```

#### Option 3 VecTree Quantization
Users can quantize a pruned and distilled 3D-GS checkpoint using the following command (default setting):
```
bash scripts/run_vectree_quantize.sh
```

#### Option 4 Overall pipline
Users can also train all pipline using the following command
```
bash runnning_codes.sh
```



## Render
Render with trajectory. By default ellipse, you can change it to spiral or others trajectory by changing to corresponding function.
```
python render_video.py --source_path PATH/TO/DATASET --model_path PATH/TO/MODEL --skip_train --skip_test --video
```
For render after the Vectree Quantization stage, you could render them through
```
python render_video.py --load_vq
```

For check rendering speed
```
python render_speed.py --source_path PATH/TO/DATASET --model_path PATH/TO/MODEL --skip_train --load_vq
```