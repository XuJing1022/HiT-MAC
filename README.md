# Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks

This repository is the official implementation of [Learning Multi-Agent Coordination for Enhancing Target Coverage in Directional Sensor Networks](https://arxiv.org/abs/2030.12345). 

![](graphicExplaining/DSN_5.Jpeg)
>📋  Project website: [HiT-MAC](https://sites.google.com/view/hit-mac)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training
To train the executor in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=-1 python main.py --env Pose-v0 --model single-att --workers 6
```

To train the coordinator in the paper, run this command:

```train
CUDA_VISIBLE_DEVICES=-1 python main.py --env Pose-v1 --model multi-att-shap --workers 6
```

## Evaluation

To evaluate my model, run:

```eval
CUDA_VISIBLE_DEVICES=-1 python main.py --env Pose-v1 --model multi-att-shap --workers 0 --load-model-dir trainedModel/best_coordinator.pth
```

You can use trained models directly from the folder "trainedModel".

## Results

Our model achieves the following performance compared with baselines:

| Model name         | Coverage Rate   | Average Gain   |
| ------------------ |---------------- | -------------- |
| MADDPG             |     85%         |      95%       |
| SQDDPG             |     85%         |      95%       |
| COMA               |     85%         |      95%       |
| ILP                |     85%         |      95%       |
| HiT-MAC            |     85%         |      95%       |


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
