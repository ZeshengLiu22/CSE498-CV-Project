# CSE498-CV-Project
Course Project for CSE498

## File description
In this project, we tested three image super-resolution methods(ESRGAN, SwinFIR, DRCT) on two post-disaster datasets. The file layout are the following:

**Data_Preprocessing/**: It includes an "LR_generation.py" file to generate our training dataset, which contains paired low-resolution and high-resolution images.

**DRCT/**: All the files for DRCT model

**SwinFIR/**: All the files for SwinFIR model

**ESRGAN/**: All the files for ESRGAN model

**TrainingResults/**: Training logs and config files.

## How to prepare the dataset:
Use the LR_generation.py, modify lines 68-74 based on your own dataset name and path, then run:

```
python LR_generation.py
```

## How to run each model:

### DRCT:

Run the following code to set the environment:

```
conda create --name drct python=3.8 -y
conda activate drct
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd DRCT
pip install -r requirements.txt
python setup.py develop
```

Then, put the provided log file under "options/train/". Put datasets or the soft links of datasets in the "datasets" folder, then modify the data path in the given config yml file.

To train the network from scratch, run:
```
python drct/train.py -opt options/train/train_DRCT_SRx4_floodnet_512.yml --launcher none 
python drct/train.py -opt options/train/train_DRCT_SRx4_floodnet_1024.yml --launcher none 
python drct/train.py -opt options/train/train_DRCT_SRx4_rescuenet_512.yml --launcher none 
python drct/train.py -opt options/train/train_DRCT_SRx4_rescuenet_1024.yml --launcher none 
```
then the saved model weights and training log can be found under "experiments" folder.

To test the trained network on test dataset, make sure to modify the path in the given .yml config file, put it under "options/test/", then run:

```
python drct/test.py -opt options/test/DRCT_SRx4_floodnet_512.yml
python drct/test.py -opt options/test/DRCT_SRx4_floodnet_1024.yml
python drct/test.py -opt options/test/DRCT_SRx4_rescuenet_512.yml
python drct/test.py -opt options/test/DRCT_SRx4_rescuenet_1024.yml

```

You can find image visualizations and test log file under "results"

### SwinFIR:

Run the following code to set the environment:
```
cd SwinFIR
pip install -r requirements.txt
python setup.py develop
```

Then, put the provided log file under "options/train/". Put datasets or the soft links of datasets in the "datasets" folder, then modify the data path in the given config yml file.

To train the network from scratch, run:
```
python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx4_from_scratch_floodnet_512.yml --launcher none
python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx4_from_scratch_floodnet_1024.yml --launcher none
python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx4_from_scratch_rescuenet_512.yml --launcher none
python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx4_from_scratch_rescuenet_1024.yml --launcher none

```
then the saved model weights and training log can be found under "experiments" folder.

To test the trained network on test dataset, make sure to modify the path in the given .yml config file, put it under "options/test/", then run:

```
python swinfir/test.py -opt options/test/SwinFIR/SwinFIR_SRx4_floodnet_512.yml
python swinfir/test.py -opt options/test/SwinFIR/SwinFIR_SRx4_floodnet_1024.yml
python swinfir/test.py -opt options/test/SwinFIR/SwinFIR_SRx4_rescuenet_512.yml
python swinfir/test.py -opt options/test/SwinFIR/SwinFIR_SRx4_rescuenet_1024.yml
```

You can find image visualizations and test log file under "results"

### ESRGAN

Adjust "datasets.py" for your dataset path.
Then run
```
python esrgan.py --dataset_name <dataset-name> --batch_size <batch_size> --hr_height <hr_height> --hr_width <hr_width>
```
<dataset-name>: FloodNet or RescueNet
<batch_size>: 4 if 128->512; 1 if 256->1024
<hr_height> <hr_width>: 512 or 1024

Adjust the "eval.py" for saved model path and dataset. Then run

## Expected results

Table: Image super-resolution results on FloodNet

| Dataset                 | Image Size   | Metrics | ESRGAN  | DRCT    | SwinFIR |
| ----------------------- | ------------ | ------- | ------- | ------- | ------- |
| **FloodNet-Validation** | **128→512**  | PSNR    | 20.3364 | 29.4241 | 29.4903 |
|                         |              | SSIM    | 0.5801  | 0.6542  | 0.6562  |
|                         | **256→1024** | PSNR    | 11.0390 | 28.5391 | 28.5886 |
|                         |              | SSIM    | 0.1980  | 0.6333  | 0.6349  |
| **FloodNet-Test**       | **128→512**  | PSNR    | 20.2959 | 29.3774 | 29.4191 |
|                         |              | SSIM    | 0.5461  | 0.6421  | 0.6437  |
|                         | **256→1024** | PSNR    | 11.2342 | 28.7146 | 28.7588 |
|                         |              | SSIM    | 0.2059  | 0.6367  | 0.6383  |

Table: Image super-resolution results on RescueNet

| Dataset               | Image Size   | Metrics | ESRGAN  | DRCT    | SwinFIR |
| --------------------- | ------------ | ------- | ------- | ------- | ------- |
| **Rescue-Validation** | **128→512**  | PSNR    | 14.3396 | 28.1367 | 28.2643 |
|                       |              | SSIM    | 0.4607  | 0.7328  | 0.7373  |
|                       | **256→1024** | PSNR    | 5.0902  | 27.8947 | 28.0207 |
|                       |              | SSIM    | 0.0165  | 0.7281  | 0.7336  |
| **Rescue-Test**       | **128→512**  | PSNR    | 14.0631 | 28.1793 | 28.2856 |
|                       |              | SSIM    | 0.4434  | 0.7281  | 0.7324  |
|                       | **256→1024** | PSNR    | 4.9545  | 27.4051 | 27.5448 |
|                       |              | SSIM    | 0.0170  | 0.7201  | 0.7264  |


