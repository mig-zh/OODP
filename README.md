# Object-Oriented Dynamics Predictor (OODP)
This repository implements the main algorithm of OODP described in the following paper:
  * Guangxiang Zhu, Zhiao Huang, Chongjie Zhang, **"Object-Oriented Dynamics Predictor**",
    _In Advances in Neural Information Processing Systems (NIPS)_, 2018.

# Requirements
- TensorFlow >= 1.4.0
- gym >= 0.9.4
- numpy
- cv2
- imp

# Training
The following command shows how to train OODP.

```
$ python runoodp.py
```

# How to show results
After training, the test results for generalization will be directly printed in the screen and the folder `results/` will be created. This folder contains the checkpoint files, summary files, and the figures plotting the learning curve (`cost.jpg` plots the loss function, and `groundtruth_MotionLoss_train.jpg` and `groundtruth_MotionLoss_test.jpg` plots RMSEs between predicted and groundtruth motions in training and unseen environments).

To show the masked images in the training or unseen environments, run `tensorboard` with these commands:

```
$ cd results/
$ tensorboard --logdir=summary_train
$ tensorboard --logdir=summary_test
```

Note that we have already provided the checkpoint file containing a model pre-trained with default parameters in `results/`, so you can directly run the above commands before training.

# Extension
OODP presents an object-oriented learning framework, which decomposes the environment into objects and predicts the dynamics of objects conditioned on both actions and object-to-object relations. This framework is the fundamental building block for generalizable and interpretable dynamics learning and can be further extended to broad domains by relaxing some assumptions. For example, MAOP extends OODP to more complex environments with a dynamic background, as described in the following paper.
  * Anonymous, **"Object-Oriented Model Learning through Multi-Level Abstraction**", under review,
    _International Conference on Learning Representations (ICLR)_, 2019.
