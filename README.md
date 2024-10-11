### Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications

!["Screenshot..."](images/mdn_preview_eccv_imptc.png "Screenshot...")

This is the official implementation for: _**Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications**_. 
Paper: [[arXiv](https://arxiv.org/abs/2410.06905)] [[ResearchGate](https://www.researchgate.net/publication/384778609_Reliable_Probabilistic_Human_Trajectory_Prediction_for_Autonomous_Applications)]

**ECCV 2024 UnCV-Workshop: Best-Paper-Award** :trophy: 

_**Abstract**_ -- Autonomous systems, like vehicles or robots, require reliable, accurate, fast, resource-efficient, scalable, and low-latency trajectory predictions to get initial knowledge about future locations and movements of surrounding objects for safe human-machine interaction. Furthermore, they need to know the uncertainty of the predictions for risk assessment to provide safe path planning. This paper presents a lightweight method to address these requirements, combining Long Short-Term Memory and Mixture Density Networks. Our method predicts probability distributions, including confidence level estimations for positional uncertainty to support subsequent risk management applications and runs on a low-power embedded platform. We discuss essential requirements for human trajectory prediction in autonomous vehicle applications and demonstrate our method's performance using multiple traffic-related datasets. Furthermore, we explain reliability and sharpness metrics and show how important they are to guarantee the correctness and robustness of a model's predictions and uncertainty assessments. These essential evaluations have so far received little attention for no good reason. Our approach focuses entirely on real-world applicability. Verifying prediction uncertainties and a model's reliability are central to autonomous real-world applications.


    @article{mdn,
        title={Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications},
        author={Hetzel, M. and Reichert, H. and Doll, K. and Sick, B.},
        journal={European Conference On Computer Vision (ECCV)},
        year={2024},
        organization={Springer},
        doi={tba}
    }


!["Screenshot..."](images/mdn_preview_eccv_ind.png "Screenshot...")
---
### Table of contents:
* [Overview](#overview)
* [Requirements](#requirements)
* [Datasets](#datasets)
* [Pretrained Models](#pretrained)
* [Data Preprocessing](#prepro)
* [Training](#training)
* [Evaluation](#evaluation)
* [License](#license)

---
<a name="overview"></a>
### Overview
This repository contains all code, data and descriptions for the Paper: "Reliable Probabilistic Human Trajectory Prediction for Autonomous Applications". The following chapters describe the necessary requirements to run our method, where to download our training and evaluation data, where to download pre-trained models and how to run the code for yourself.


---
<a name="requirements"></a>
### Requirements

The framework uses the following system configuration. The exact python requirements can be found in the corresponding requirements.txt file.

```
# Software
Ubuntu 22.04 LTS
Python 3.10
Pytorch 2.1.0
CUDA 12
CuDNN 8.9
```

**Docker:**
If you want to use docker, we recommend the Nvidia NGC Pytorch 23.08 image: [[Info]](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-08.html) [[Docker Image]](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch). Run the image and install the corresponding python requirements.txt and your ready to go.


---
<a name="datasets"></a>
### Datasets
We use pedestrian trajectory data from four popular road traffic datasets for our methods training and evaluation (NuScenes, Waymo, inD and IMPTC), as well as the ETH/UCY surveillance dataset for cross-comparison with different Human Trajectory Prediction (HTP) frameworks. For all datasets we extracted the raw pedestrians trajectories, applied a human-ego-centric coordinate system transformation as pre-processing, and subsampled the input- and output horizon data for training and evaluation. The four traffic related datasets are resampled to fixed 10.0 Hz sampling rate. The ETH/UCY dataset is untouched with a 2.5 Hz sampling rate. Additional information (Papers/Code/Data) about all used datasets can be found below:

**IMPTC:** [[Paper]](https://ieeexplore.ieee.org/document/10186776) [[Data]](https://github.com/kav-institute/imptc-dataset)


**inD:** [[Paper]](https://ieeexplore.ieee.org/document/9304839) [[Data]](https://github.com/ika-rwth-aachen/drone-dataset-tools)


**nuScenes:** [[Paper]](https://arxiv.org/abs/1903.11027) [[Data]](https://github.com/nutonomy/nuscenes-devkit#nuscenes-setup)


**Waymo:** [[Paper]](https://arxiv.org/abs/1912.04838) [[Data]](https://waymo.com/intl/en_us/open/download/)


**ETH/UCY:** [[Paper]](https://ieeexplore.ieee.org/document/5459260) [[Data]](https://github.com/StanfordASL/Trajectron-plus-plus/tree/master/experiments/pedestrians/raw/raw/all_data)


---
<a name="pretrained"></a>
### Pre-trained Models and Datasets
Pre-trained models and pre-processed training and evaluation data is provided within the table below.

| Dataset       | Models | Data | Status    |
|:-------------:|:---------------:|:-------------:|:---------:|
| All           | [[Download]](https://drive.google.com/file/d/1K3bcl-R-9K1TyH-e5ZiC8ljnH5UKZTqn/view?usp=drive_link)             | [[Download]](https://drive.google.com/file/d/17itbH_ufwHuAJwPv70KvXR1qrtwDcATA/view?usp=drive_link)          | $${\color{green}online}$$ |


---
<a name="prepro"></a>
### Data Preprocessing
We extracted all pedestrian trajectories from NuScenes, Waymo, inD and IMPTC datasets. For NuScenes, Waymo, and IMPTC we adopt the pre-defined train/test/eval data splits. For inD such splits do not exists by default, therefore we defined our own (Train: sequences 00-06 and 18-29, Test: sequences 07-17, Eval: sequences 30-32). For homogenity NuScenes is up-sampled from 2.0 to 10.0 Hz, inD and IMPTC are down-sampled from 25.0 to 10.0 Hz. Waymo is untouched with 10.0 Hz sampling rate by default. To overcome coordinate system related biases or dependencies we transfered all trajectories into an independet human-ego-centric coordinate system with the position at the last input timestep is at position (0.0, 0.0), and all other input or ground truth positions are transfered related to this origin. We adapt the commonly used trajectory splitting into 3.2 s input horizon and 4.8 s forecast horizon. Longer trajectories are splitted into multiple subsplits with a step of 1 second. As a result the different dataset contains the following amount of trajectories.

| Dataset       | Train             | Test            | Eval |
|:-------------:|:-----------------:|:---------------:|:----:|
| IMPTC         | 189595            | 56694           | 19148|
| inD           | 166657            | 26300           | 28851|
| NuScenes      | 105456            | 28718           | 68419|
| Waymo         | 231091            | 22397           | 17454|


---
<a name="training"></a>
### Model Training
To run a training you can use the following command. All relevant information are provided by the configuration file. It contains all necessary paths, parameters and configurations for training and evaluation. For every dataset type one can create unlimited different configuration files.
```
# Start a training using IMPTC dataset and default config
train.py --target=imptc --configs=default_peds_imptc.json --gpu=0 --log --print

# Arguments:
--target: target dataset
--configs: target dataset specific config
--gpu: gpu id to be used for the training
--log: write training progress and feedback to log file
--print: show training progress and feedback in console
```


---
<a name="evaluation"></a>
### Model Evaluation
To run a model evaluation you can use the following command. The evaluation shares the config file with the training script.
```
# Start an evaluation for an IMPTC dataset trained model
testing.py --target=imptc --configs=default_peds_imptc.json --gpu=0 --log --print

# Arguments:
--target: target dataset
--configs: target dataset specific config
--gpu: gpu id to be used for the evaluation
--log: write evaluation results to log file
--print: show evaluation results in console
```

**Pre-trained models:**
If you want to use a pre-trained model you must locate them into the correct subfolder structure: Starting at the defined "result_path" from the configuration file: 

```
# Structure:
"<result_path>/base_mdn/<target>/<config-file-name>/checkpoints/model_final.pt"

# Example for IMPTC:
"<result_path>/base_mdn/imptc/default_peds_imptc/checkpoints/model_final.pt".
```

---
<a name="license"></a>
## License:
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details

---

## Contributors:

<a href="https://github.com/kav-institute/mdn_trajectory_forecasting/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kav-institute/mdn_trajectory_forecasting" />
</a>
