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




## Expected results