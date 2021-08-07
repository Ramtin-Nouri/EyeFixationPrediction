# [Eye Fixation Prediction](https://github.com/Ramtin-Nouri/EyeFixationPrediction)
This is an implementation of an eye fixation prediction model, build for the Uni Hamburg Computer Vision II project.  

It is build using TF2-**Keras**. For this I also build a more generic template (integrated as a submodule) to start of Keras projects more quickly.  
The architecture and details about the model can be read in the paper.  

The dataset provided for this project consists of 3006 training samples, 1128 validation samples and 1032 test images with held-out ground truth by the project instructors. The provided dataset is based on the [DUT-OMRON](http://saliencydetection.net/dut-omron/) dataset.  

I use the following additional datasets: 
- [MIT](http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html)
- [CAT2000](http://saliency.mit.edu/results_cat2000.html)
- [SALICON](http://salicon.net/download/)
