# DR-GAN(pytorch implement for DR-GAN)
## paper link
http://cvlab.cse.msu.edu/pdfs/Tran_Yin_Liu_CVPR2017.pdf
## requirements
 - python3.6
 - pytorch0.3.1
 - torchvision
 - numpy
 - opencv-python
 - pillow
## usage
 - view option/base_option.py train_option.py test_option.py and revise setting yourself
 - train: python train.py
 - test: python test.py
 - I add some data augment in test.py for experiment comparsion, they are origin, gauss_noise, 
   gamma_transform, shadow, random color and blur consecutively.
 - pretrained model in directory checkpoints
## dataset 
   I train this model use cfp-dataset, due to dataset size and lack of labeled information, my GAN can't converge very well. If you have 
  other dataset like Multi-PIE it would be helpful.
## more details
   You can read some comments in code or experiment_record.md for more details
  
**If you have any questions please come up with issue**
