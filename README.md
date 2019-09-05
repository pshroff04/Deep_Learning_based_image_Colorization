# Image Colorization
This repo is the pytorch implementation of paper [Colorful Image Colorization][dill]. This implementation is sightly different than the paper.

### Clone this repository

  - git clone [https://github.tamu.edu/prateek-shroff/DL_Colorization.git][git]

### Dependencies
 - pytorch 0.3+ and torchvision 0.1
 - python 3.5
 - Basic packages (numpy, scipy, matplotlib, tensorboardx)

### Training
Start from scratch
```
python main.py --batch_size 32 --gpu 0,1 --num_iterations 40000 --save_directory <path/to save weights>
```
Or
Resume from a checkpoint
```
python main.py --resume <iteration_number>
```
### Inference

```
python main.py --infer_iter <which iteration weight to pick for inference> --weight_dir <path to saved weight> 
```

   [dill]: <https://arxiv.org/pdf/1603.08511.pdf>
   [git]: <https://github.tamu.edu/prateek-shroff/DL_Colorization>


