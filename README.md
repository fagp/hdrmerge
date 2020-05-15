This repo have the implementation for a CNN-based approach to learn the transformation from raw images to merge/final hdr image.

Requirements:
-torch
-torchvision
-visdom
-rawpy
-scipy
-scikit-image

Dataset configuration
This repo uses Google HDR+ dataset (https://hdrplusdata.org). For setting the path to the downloaded dataset modify "defaults/dataconfig_test.json" and "defaults/dataconfig_train.json" files.
Example:
"path_raw": "hdrplus/20171106_subset/bursts",
"path_result": "hdrplus/20171106_subset/results_20171023",

For setting the burst size modify the key "size_burst" within "defaults/dataconfig_test.json" and "defaults/dataconfig_train.json" (i.e. "size_burst":"3",). Also, you need to alter accordingly the number of input channels for the model. This is done by changing "in_channels" key within "defaults/modelconfig.json". (i.e. "in_channels":"3",)

An example line for learning merge is:
python train.py --experiment=my_train --loss=mse --dataset=hdrmerge --model=unetmerge --use_cuda=0 --optimizer=Adam --optimizerparam="{'lr':'0.00001'}" --visdom --train_worker=4 --test_worker=4 --batch_size=1

An example line for learning the final hdr image is:
python train.py --experiment=my_train --loss=mse --dataset=hdrfinal --model=unetfinal --use_cuda=0 --optimizer=Adam --optimizerparam="{'lr':'0.00001'}" --visdom --train_worker=4 --test_worker=4 --batch_size=1

--dataset parameter allows one of the following [hdrmerge, hdrfinal].
--loss parameter allows one of the following [mse, l1, nmsepad]
--model parameter allows one of the following [unetmerge, unetfinal]
--optimizer parameter allows one of the following [Adam, SGD, RMSprop]

For a complete list of allowed parameters type: python train.py -h. Checkpoints will be saved in ../out/<experiment>, where "experiment" is the name you defined.

Note: The parameter --imsize is deprecated and is only kept for compatibility reasons. It will be removed in future versions.

