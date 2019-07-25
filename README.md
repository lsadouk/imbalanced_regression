# A cost-sensitive learning approach and evaluation strategies for handling regression under imbalanced domains

Welcome. This repository contains the data and scripts comprising the 'A Novel Cost-sensitive Algorithm and New Evaluation Strategies for Regression in Imbalanced Domains'. 

Included are the tools to allow you to easily run the code.

This readme is a brief overview and contains details for setting up and running the project. Please refer to the following:

<h1>Running the code</h1><br/>
<h2>Initial requirements</h2>

1. To the code, the environment needed is Matlab. So you need to install Matlab.
2. The MatConvNet Toolbox is needed in this project. It has already been downloaded and compiled for you. So, you don't need to install and compile MatConvNet. But, if you have your own version of MatConvNet, you can do so by replacing the MatConvNet folder within 'imbalanced_regression' directory by your own.

<h2>Usage</h2>
There are several use cases for this project:

1. You can train and test the the neural network by running the file 'proj_regression.m' (<b>see examples below </b>):
- using the L2 Loss function, the probabilistic loss function with the normal distribution relevance (NDR), the probabilistic loss function with the Kernel distribution relevance (KDR), or the Weighted loss function with the Kernel distribution relevance
- using no resampling strategy (n), the under-sampling technique (u), or the over-sampling technique (o), or the Smoter technique with o2_u0.5 (s)
- on one of the following datasets: (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) 

2. In order to visualize and compare between the following techniques (experiments): 
     * <b>classical method</b>: training an MLP with no resampling strategy and using the l2 loss function (denoted as “l<sub>2</sub> Unb.” in the article),
     * <b>probabilistic method</b>: training an MLP with no resampling strategy and using the lp probabilistic loss function (with either normal or kernel distribution) (referred to as “l<sub>p</sub>  Unb.” in the article),
     * <b>undersampling method</b>: training an MLP with the undersampling strategy and using the l2 loss function (denoted as “l<sub>2</sub>  Bal<sub>u</sub>” in the article),
     * <b>oversampling method</b>: training an MLP with the oversampling strategy and using the l2 loss function (denoted as “l<sub>2</sub>  Bal<sub>o</sub>” in the article),
     * <b>SmoteR method</b>: training an MLP with the SmoteR strategy and using the l2 loss function (denoted as “l<sub>2</sub>  Bal<sub>smoter</sub>” in the article),
     * <b>weighted method</b>: training an MLP with the weighting strategy based on the cost-sensitive loss function of Wang 1999 ("Wan, C., Wang, L., & Ting, K. (1999b). Introducing cost-sensitive neural networks. In Proceedings of the second international conference on information, communications and signal processing (pp. 1–4).) (referred to as “l<sub>weighted</sub>  Unb.” in the article),

you can choose one of the evalutation measures below (scalar or graphical-based):
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
     * Please select the method for handling imbalanced data (o)data pre-processing: Oversampling, (u)data pre-processing: Undersampling,(s)data pre-processing: Smoter (o2_u0.5), (n)nothing  n
     * Please enter the loss (0)L2 loss, (1)P loss w. NDR, (2)P loss w. KDR, (3) Weighted loss w. KDR 2
     * Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing)  10
     * Please select the dataset (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) abalone
     * Please choose the performance index: (mae)MAE / (rmse)RMSE /(w)Weighted MAE/(tgm)GME/(tcwa)CWE/(wm)WMAPE/(tm)Threshold MAPE tgm

<i>The displayed result is :</i>
> Lowest tgm error is 1.891 at epoch 30

PS: Note that 30 stands for the epoch having the recorded lowest error.

The code :
- outputs the lowest test error (in terms of the chosen scalar measure).
- displays a plot of the objective function and error per epoch for both training and testing sets.
- displays the weights of the 1st convolutional layer filters.

<h3>2. Example of training and testing our the oversampling algorithm and evaluating it with a scalar measure</h3>
In this example, we want to train the oversampling algorithm on a regression task using the "abalone" dataset. The scalar measure used for evaluation is GME.

To do so, follow these steps:
1. run proj_regression.m
2. select the following:
     * Please select the method for handling imbalanced data (o)data pre-processing: Oversampling, (u)data pre-processing: Undersampling,(s)data pre-processing: Smoter (o2_u0.5), (n)nothing  o
     * Please enter the loss (0)L2 loss, (1)P loss w. NDR, (2)P loss w. KDR, (3) Weighted loss w. KDR 0
     * Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing)  10
     * Please select the dataset (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) abalone
     * Please choose the performance index: (mae)MAE / (rmse)RMSE / (w)Weighted MAE / (tgm)GME / (tcwa)CWE / (wm)WMAPE / (tm)Threshold MAPE 
     * Please select the undersampling rate 0.5, 1(default), 2: 1

<i>The displayed result is :</i>
> Lowest tgm error is 1.925 at epoch 30

<h3>3. Example of training and testing our cost-sensitive learning regression algorithm and evaluating it with a graphical measure</h3>
In this example, we want to train our algorithm on a regression task using the "accel" dataset. The measure used is the graphical measure REC<sub>G-Mean</sub> curve in which the REC<sub>G-Mean</sub> curves of the undersampling technique l<sub>2</sub> Bal<sub>u</sub>, the oversampling technique l<sub>2</sub> Bal<sub>o</sub>, the SmoteR technique l<sub>2</sub> Bal<sub>smoter</sub>, the classical method l<sub>2</sub> Bal<sub>n</sub>, our cost-sensitive technique l<sub>p</sub> Bal<sub>n</sub>, and the weighted technique l<sub>weighted</sub> Bal<sub>n</sub> are displayed.

To do so, follow these steps:
1. Follow the steps of Example 1 in order to get test set predicted outputs for the cost-sensitive technique l<sub>p</sub> Bal<sub>n</sub>, which will be saved in "result_test_data/result_abalone_r_L2_n.mat".
2. Follow the steps of Example 1 except, after running the code,  choose "3" instead of "2" in the entry 'Please enter the loss'), in order to get test set predicted outputs for the cost-sensitive technique l<sub>weighted</sub> Bal<sub>n</sub>, which will be saved in "result_test_data/result_abalone_r_L3_n.mat".
3. Follow the steps of Example 2 to get test set predicted outputs for the oversampling technique l<sub>2</sub> Bal<sub>o</sub>, which  will be saved in  "result_test_data/result_abalone_r_L0_o.mat".
4. Follow the steps of Example 2 to get test set predicted outputs for the oversampling technique l<sub>2</sub> Bal<sub>smoter</sub>, which  will be saved in  "result_test_data/result_abalone_r_L0_s.mat".
5. Follow the steps of Example 2 except (i) in the code, change the number of epochs as follows: opts.numEpochs =  140; and (ii) after running the code,  choose "u" instead of "o" in the entry 'method for handling imbalanced data'). As a result, test set predicted outputs for the undersampling technique l<sub>2</sub> Bal<sub>u</sub>, are obtained and saved in "result_test_data/result_abalone_r_L0_u.mat".
6. Follow the steps of Example 2 except (i) in the code, change the number of epochs as follows: opts.numEpochs =  100; and (ii) after running the code,  choose "n" instead of "o" in the entry 'method for handling imbalanced data'). As a result, test set predicted outputs for the classical method l<sub>2</sub> Bal<sub>n</sub> are obtained and saved in "result_test_data/result_abalone_r_L0_n.mat".

7. Go to the 'REC' folder and run rec_GMean_CWA.m

The code :
- displays a plot of REC<sub>TPR</sub> curves of different techniques and outputs their AOCs.
- displays a plot of REC<sub>TNR</sub> curves of different techniques and outputs their AOCs.
- displays a plot of REC<sub>G-Mean</sub> curves of different techniques.

PS: These plots are also saved in the "result_REC_plots" directory.
 
<i>The displayed result is:</i>
> 1. Computing AOCs of the TNR RECs: 
>      * l_2 Unb. AOC = 1.735718
>      * l_2 Bal_u AOC = 1.840080
>      * l_2 Bal_o AOC = 1.995475
>      * l_2 Bal_smoter AOC = 2.217166
>      * l_weighted Unb. AOC = 2.026801
>      * l_P Unb.  AOC = 1.779982
> 2. Computing AOCs of the TPR RECs: 
>      * l_2 Unb. AOC = 5.106965
>      * l_2 Bal_u AOC = 3.571501
>      * l_2 Bal_o AOC = 3.182202
>      * l_2 Bal_smoter AOC = 3.218911
>      * l_weighted Unb. AOC = 4.503599
>      * l_P Unb.  AOC = 3.463429
> 3. Computing AOCs of GMean RECs for (1)l_2 Unb.,(2)l_2 Bal_u,(3)l_2 Bal_o,(4)l_2 Bal_smoter,(5)l_weighted Unb.,(6)Our l_P Unb.: 
>      * AOC of 1 = 3.689048
>      * AOC of 2 = 2.806411
>      * AOC of 3 = 2.654684
>      * AOC of 4 = 2.764059
>      * AOC of 5 = 3.510593
>      * AOC of 6 = 2.744590

