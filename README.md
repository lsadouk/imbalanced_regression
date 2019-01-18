# A cost-sensitive learning approach and evaluation strategies for handling regression under imbalanced domains

Welcome. This repository contains the data and scripts comprising the 'A Novel Cost-sensitive Algorithm and New Evaluation Strategies for Regression in Imbalanced Domains'. 

Included are the tools to allow you to easily run the code.

This readme is a brief overview and contains details for setting up and running TFF with PLF. Please refer to the following:

<h1>Running the code</h1><br/>
<h2>Initial requirements</h2>

1. To the code, the environment needed is Matlab. So you need to install Matlab.
2. The MatConvNet Toolbox is needed in this project. It has already been downloaded and compiled for you. So, you don't need to install and compile MatConvNet. But, if you have your own version of MatConvNet, you can do so by replacing the MatConvNet folder within 'imbalanced_regression' directory by your own.

<h2>Usage</h2>
There are several use cases for this project:

1. You can train and test the the neural network by running the file 'proj_regression.m' (<b>see examples below </b>):
- using the L2 Loss function (lambda=0), the probabilistic loss function with the normal distribution (lambda=1), or the probabilistic loss function with the normal distribution (lambda=2)
- using no resampling strategy (n), the under-sampling technique (u), or the over-sampling technique (o)
- on one of the following datasets: (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) 

2. In order to visualize and compare between the following 4 techniques: 
     * <b>classical method</b>: training an MLP with no resampling strategy and using the l2 loss function (denoted as “l<sub>2</sub> Unb.” in the article),
     * <b>probabilistic method</b>: training an MLP with no resampling strategy and using the lp probabilistic loss function (with either normal or kernel distribution) (referred to as “l<sub>p</sub>  Unb.” in the article),
     * <b>undersampling method</b>: training an MLP with the undersampling strategy and using the l2 loss function (denoted as “l<sub>2</sub>  Bal<sub>u</sub>” in the article),
     * <b>oversampling method</b>: training an MLP with the oversampling strategy and using the l2 loss function (denoted as “l<sub>2</sub>  Bal<sub>o</sub>” in the article).

, you can choose one of the evalutation measures below (scalar or graphical-based):
- <b>Scalar measures</b>: Mean Absolute Error(mae), Root Mean Squared Error (rmse), Geometric-Mean Error GME (tgm), Class-Weighted Error CWE (tcwa) (<b>see example 1 and 2 below</b>)
- <b>Graphical-based measures</b>: 
     * REC<sub>G-Mean</sub> curve ==> The REC<sub>TPR</sub>, REC<sub>TNR</sub>, and REC<sub>G-Mean</sub> curves are all displayed, and their corresponding AOCs for each of the 4 techniques are displayed  (<b>see example 1 and 2 below</b>)
     * REC<sub>CWA</sub> curve ==> The REC<sub>TPR</sub>, REC<sub>TNR</sub>, and REC<sub>CWA</sub> curves are all displayed, and their corresponding AOCs for each of the 4 techniques are displayed (<b>see example 3 below</b>)

<h2>Examples for training and/or testing our models : </h2>
<h3>1. Example of training and testing our cost-sensitive learning regression algorithm and evaluating it with a scalar measure</h3>
In this example, we want to train our cost-sensitive learning algorithm on a regression task using the "abalone" dataset. The scalar measure used for evaluation is GME.

To do so, follow these steps:
1. run proj_regression.m
2. select the following:
     * Please select the method for handling imbalanced data (o)data pre-processing: Oversampling, (u)data pre-processing: Undersampling, (n)nothing  n
     * Please enter the loss (0)L2 loss, (1)P loss w. normal distribution, (2)P loss w. kernel distribution 2
     * Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing)  10
     * Please select the dataset (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) abalone
     * Please choose the performance index: (mae)MAE / (rmse)RMSE /(w)Weighted MAE/(tgm)GME/(tcwa)CWE/(wm)WMAPE/(tm)Threshold MAPE tgm

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
     * Please select the method for handling imbalanced data (o)data pre-processing: Oversampling, (u)data pre-processing: Undersampling, (n)nothing  o
     * Please enter the loss (0)L2 loss, (1)P loss w. normal distribution, (2)P loss w. kernel distribution 0
     * Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing)  10
     * Please select the dataset (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) abalone
     * Please choose the performance index: (mae)MAE / (rmse)RMSE / (w)Weighted MAE / (tgm)GME / (tcwa)CWE / (wm)WMAPE / (tm)Threshold MAPE 
     * Please select the undersampling rate 0.5, 1(default), 2: 1

<b>The displayed result is :</b>
Lowest tgm error is 1.891 (37)
> Note that 37 stands for the epoch having the recorded lowest error.

<h3>3. Example of training and testing our cost-sensitive learning regression algorithm and evaluating it with a graphical measure</h3>
In this example, we want to train our algorithm on a regression task using the "abalone" dataset. The measure used is the graphical measure REC<sub>G-Mean</sub> curve in which the REC<sub>G-Mean</sub> curves of the undersampling technique l<sub>2</sub> Bal<sub>u</sub>, the oversampling technique l<sub>2</sub> Bal<sub>o</sub>, the classical method l<sub>2</sub> Bal<sub>n</sub>, and our cost-sensitive technique l<sub>p</sub> Bal<sub>n</sub> are displayed.

To do so, follow these steps:
1. Follow the steps of Example 1 in order to get test set predicted outputs for the cost-sensitive technique l<sub>p</sub> Bal<sub>n</sub>, which will be saved in "result_test_data/result_abalone_r_L2_n.mat".
2. Follow the steps of Example 2 to get test set predicted outputs for the oversampling technique l<sub>2</sub> Bal<sub>o</sub>, which  will be saved in  "result_test_data/result_abalone_r_L0_o.mat".
3. Follow the steps of Example 2 except (i) in the code, change the number of epochs as follows: opts.numEpochs =  140; and (ii) after running the code,  choose "u" instead of "o" in the entry 'method for handling imbalanced data'). As a result, test set predicted outputs for the undersampling technique l<sub>2</sub> Bal<sub>u</sub>, are obtained and saved in "result_test_data/result_abalone_r_L0_u.mat".
4. Follow the steps of Example 2 except (i) in the code, change the number of epochs as follows: opts.numEpochs =  100; and (ii) after running the code,  choose "n" instead of "o" in the entry 'method for handling imbalanced data'). As a result, test set predicted outputs for the classical method l<sub>2</sub> Bal<sub>n</sub> are obtained and saved in "result_test_data/result_abalone_r_L0_n.mat".
4. Go to the 'REC' folder and run rec_GMean_CWA.m

The code :
- displays a plot of REC<sub>TPR</sub> curves of different techniques and outputs their AOCs.
- displays a plot of REC<sub>TNR</sub> curves of different techniques and outputs their AOCs.
- displays a plot of REC<sub>G-Mean</sub> curves of different techniques.

> PS: These plots are also saved in the "result_REC_plots" directory.
 
The displayed result is:
1. Computing AOCs of the TNR RECs: 
     * l_2 Unb. AOC = 1.144191
     * l_2 Bal_u AOC = 2.764341
     * l_2 Bal_o1 AOC = 1.619792
     * l_P Unb. AOC = 1.860168
2. Computing AOCs of the TPR RECs: 
     * l_2 Unb. AOC = 3.643603
     * l_2 Bal_u AOC = 2.386789
     * l_2 Bal_o1 AOC = 2.300064
     * l_P Unb. AOC = 2.273694
3. Computing AOCs of GMean RECs for (1)l_2 Unb., (2)l_2 Bal_u, (3)l_2 Bal_o, (4)l_P Unb.: 
     * AOC of 1 = 2.620863
     * AOC of 2 = 2.601916
     * AOC of 3 = 1.973410
     * AOC of 4 = 2.071313

