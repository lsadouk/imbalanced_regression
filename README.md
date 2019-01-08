# A cost-sensitive learning approach and evalutation strategies for handling regression under imbalanced domains

Welcome. This repository contains the data and scripts comprising the 'A Novel Cost-sensitive Algorithm and New Evaluation Strategies for Regression in Imbalanced Domains'. 

Included are the tools to allow you to easily run the code.

This readme is a brief overview and contains details for setting up and running TFF with PLF. Please refer to the following:

<h1>Running the code</h1><br/>
<h2>Initial requirements</h2>

1. To the code, the environment needed is Matlab. So you need to install Matlab.
2. The MatConvNet Toolbox is needed in this project. It has already been downloaded and compiled for you. So, you don't need to install and compile MatConvNet. But, if you have your own version of MatConvNet, you can do so by replacing the MatConvNet folder within 'traffic_flow_code_CNN/' directory  by your own.

<h2>Usage</h2>
There are several use cases for this project:

1. You can train and test the cost-sensitive learning approach by running the file 'proj_regression.m'.

You can choose to train and test the CNN (<b>see example 1 below </b>):
- among different prediction time interval (5-, 10-, 15-, and 20-min forecasting)
- with either the L2 Loss function (lambda=0) or the probabilistic loss function (lambda=1)
- on one of the following networks (freeways): H101_North_D7 / I5_North_D7 / I5_South_D7 / I5_North_D11 / I450_North_D7 / I210_West_D7 

2. You can choose to test the CNN using a an available CNN which was already trained using traffic data of 'H101_North_D7' freeway (<b>see example 3 below</b>)
If you wich to test using a CNN trained on another network (freeway), you first train the CNN using the desired network (use case 1) then call it for testing (use case 2).

3. You can train and test Deep Belief Networks (DBNs) by going into 'traffic_flow_code_DBN/examples/' directory and running the file 'proj_traffic_flow_prediction_DBN.m'

4. You can choose to test the DBN using a an available DBN which was already trained using traffic data of 'H101_North_D7' freeway (<b>see example 2 below</b>).
If you wich to test using a DBN trained on another network (freeway), you first train the CNN using the desired network (use case 3) then call it for testing (use case 4).


<b>PS. If you want to compare CNN and DBN performances with existant methods, you can try :</b>
- the Support Vector Machin (SVM) by going into 'traffic_flow_code_SVM/' directory and running the file 'proj_traffic_flow_prediction_SVM.m'
- the ARIMA by y going into 'traffic_flow_code_ARIMA/' directory and running the file 'proj_traffic_flow_prediction_ARIMA.m'
- the HW-ExpS by y going into 'traffic_flow_code_HW-exp/' directory and running the file 'main.m'

<h2>Examples for training and/or testing our models : </h2>
<h3>1. Example of training and testing our cost-sensitive learning regression algorithm and evaluating it with a scalar measure</h3>
In this example, we want to train our cost-sensitive learning algorithm on a regression task using the "abalone" dataset. The scalar measure used for evaluation is GME.

To do so, follow these steps:
1. run proj_regression.m
2. select the following:
- Please select the method for handling imbalanced data (o)data pre-processing: Oversampling, (u)data pre-processing: Undersampling, (n)nothing  n

- Please enter the loss (0)L2 loss, (1)P loss w. normal distribution, (2)P loss w. kernel distribution2

- Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing)  10

- Please select the dataset (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) abalone

- Please choose the performance index: (mae)MAE / (rmse)RMSE /(w)Weighted MAE/(tgm)GME/(tcwa)CWE/(wm)WMAPE/(tm)Threshold MAPE tgm

<b>The displayed result is :</b>
Lowest tgm error is 1.891 37

The code :
- outputs the lowest test error (in terms of the chosen scalar measure).
- displays a plot of the objective function and error per epoch for both training and testing sets.
- displays the weights of the 1st convolutional layer filters.

<h3>2. Example of training and testing our the oversampling algorithm and evaluating it with a scalar measure</h3>
In this example, we want to train the oversampling algorithm on a regression task using the "abalone" dataset. The scalar measure used for evaluation is GME.

To do so, follow these steps:
1. run proj_regression.m
2. select the following:
- Please select the method for handling imbalanced data (o)data pre-processing: Oversampling, (u)data pre-processing: Undersampling, (n)nothing  o

- Please enter the loss (0)L2 loss, (1)P loss w. normal distribution, (2)P loss w. kernel distribution 0

- Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing)  10

- Please select the dataset (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) abalone

- Please choose the performance index: (mae)MAE / (rmse)RMSE /(w)Weighted MAE/(tgm)GME/(tcwa)CWE/(wm)WMAPE/(tm)Threshold MAPE tgm

- Please select the undersampling rate 0.5, 1(default), 2: 1

<b>The displayed result is :</b>
Lowest tgm error is 1.891 37

<h3>3. Example of training and testing our cost-sensitive learning regression algorithm and evaluating it with a graphical measure</h3>
In this example, we want to train our algorithm on a regression task using the "abalone" dataset. The measure used is the graphical measure G-Mean REC curve in which the G-Mean REC curve of the undersampling technique l_2  〖Bal〗_u, the oversampling technique l_2  〖Bal〗_o, the classical method l_2  〖Bal〗_n, and our cost-sensitive technique l_p  〖Bal〗_n are displayed.

To do so, follow these steps:
1. Follow the steps of Example 1 in order to get test set predicted outputs for the cost-sensitive technique l_p  〖Bal〗_n, which are saved within the file "result_abalone_r_L2_n.mat" of the folder "result_test_data".
2. Follow the steps of Example 2 to get test set predicted outputs for the oversampling technique l_2  〖Bal〗_o, which are saved within the file "result_abalone_r_L0_o.mat" of the folder "result_test_data".
3. Follow the steps of Example 2 (except choose "u" instead of "o" in the 'method for handling imbalanced data' and change the number of epochs as follows: opts.numEpochs =  140;) to get test set predicted outputs for the undersampling technique l_2  〖Bal〗_u, which are saved within the file "result_abalone_r_L0_u.mat" of the folder "result_test_data".
4. Follow the steps of Example 2 (except choose "n" instead of "o" in the 'method for handling imbalanced data' and change the number of epochs as follows: opts.numEpochs =  100;) to get test set predicted outputs for the classical method l_2  〖Bal〗_n, which are saved within the file "result_abalone_r_L0_n.mat" of the folder "result_test_data".
4. Go to the 'REC' folder and run rec_GMean_CWA.m

The code :
- displays a plot of TPR REC curves of different techniques and outputs their AOCs.
- displays a plot of TNR REC curves of different techniques and outputs their AOCs.
- displays a plot of G-Mean REC curves of different techniques.

PS: These plots are also saved in the "result_REC_plots" directory.
 
The displayed result is:
(1)Computing AOCs of the TNR RECs: 
l_2 Unb. AOC = 1.144191
l_2 Bal_u AOC = 2.764341
l_2 Bal_o1 AOC = 1.619792
l_P Unb. AOC = 1.860168
(2)Computing AOCs of the TPR RECs: 
l_2 Unb. AOC = 3.643603
l_2 Bal_u AOC = 2.386789
l_2 Bal_o1 AOC = 2.300064
l_P Unb. AOC = 2.273694
(3)Computing AOCs of GMean RECs for (1)l_2 Unb.,(2)l_2 Bal_u,(3)l_2 Bal_o,(4)l_P Unb.: 
AOC of 1 = 2.620863
AOC of 2 = 2.601916
AOC of 3 = 1.973410
AOC of 4 = 2.071313

