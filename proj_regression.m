
function [net, info] = proj_regression()
%code by Lamyaa Sadouk, FST Settat

run matconvnet-1.0-beta16/matlab/vl_setupnn ; %run('matconvnet-1.0-beta16', 'matlab', 'vl_setupnn.m') ;

opts.learningRate = 0.001; % % TO BE CHOSEN BASED ON THE DATASET
opts.continue = true;
opts.gpus = [] ; %GPU support is off by default

%% Inputs
opts.method =input('Please select the method for handling imbalanced data (o)data pre-processing: Oversampling, (u)data pre-processing: Undersampling,(s)data pre-processing: Smoter (o2_u0.5), (n)nothing  ', 's');
opts.lambda =input('Please enter the loss (0)L2 loss, (1)P loss w. NDR, (2)P loss w. KDR, (3) Weighted loss w. KDR');
kfold =input('Please enter the k-fold (k-1 for training & 1 for testing)_(0 for testing)  ');% kfold=0 for TESTING and kfold = 10 for TRAINING
opts.dataset =input('Please select the dataset (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) ', 's');
opts.pIndex =input('Please choose the performance index: (mae)MAE / (rmse)RMSE /(w)Weighted MAE/(tgm)GME/(tcwa)CWE/(wm)WMAPE/(tm)Threshold MAPE ','s'); %deleted(g)GMRAE

%% Where trained outputs nets are saved
opts.expDir = fullfile('result_nets',strcat('data_', opts.dataset,'_r', ...
     int2str(opts.lambda),'_',opts.method,'_',opts.pIndex, '_', opts.method,'newLoss' )) ; 

%% -------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
imdb_filename = fullfile('data_preprocessed', strcat('imdb_', opts.dataset,'_r','.mat')); 
if exist(imdb_filename, 'file')
    load(imdb_filename) ; %  imdb = load(imdb_filename) ;  %save(imdb_filename, '-struct', 'imdb') ;
else
  [imdb, s_factor] = setup_data(kfold, opts.dataset);
end

%% -------------------------------------------------------------------
%                                                         Prepare model
%                                                         architecture
% --------------------------------------------------------------------
% specify the network architecture w/ cnn_init function
opts.numEpochs =  141; % TO BE CHOSEN BASED ON THE DATASET
nb_features = size(imdb.images.data,1); % # of attributes
net = cnn_init_regression(nb_features, opts.lambda);  


%% -------------------------------------------------------------------
%                                                 Prepare model
%                                                 distribution
% --------------------------------------------------------------------
%1. prepare the distribution model to be used in data-processing (if selected by the user)
%or in the euclidean loss function
%if the size of training labels is  larger than 20000, randomly select
%2000 samples to generate the distribution model

index = find(imdb.images.set==  1); % take only testing data
label = imdb.images.labels;
% pd for performance measure (index of performance)
opts.pd_model_pmeasure = fitdist(label(:),'kernel'); %,'Kernel', 'epanechnikov');
%opts.pd_model_pmeasure.BandWidth = 1; % was 1 % need to DELETE LATER
% maximum value of pdf for performance measure (index of performance)
pdf_model = pdf(opts.pd_model_pmeasure,label);
opts.pd_model_max_pmeasure = max(pdf_model);

if opts.lambda == 1 || opts.lambda == 2 || opts.lambda == 3 % normal or kernel distribution =>(NDR or KDR)
    if(length(index) > 2000) % was 20000
        label = imdb.images.labels(index);
        randNdx=randperm(length(label)); 
        label = label(randNdx);
        label = label(1:2000); % reduce data from 515345 to 50000 to speed up the pdf process        
    end % else do nothing
end

if opts.lambda == 2 || opts.lambda == 3 % kernel distribution => KDR
    pd_model = fitdist(label(:),'kernel'); % options:'Kernel','epanechnikov'
    pdf_model = pdf(pd_model,label);
    pd_model_max = max(pdf_model);
    if opts.lambda == 2
        weighting_type = 'addition'; % our P loss
    else %opts.lambda == 3
        weighting_type = 'multiplication'; % weighted P loss
    end
elseif opts.lambda == 1 % normal distribution with KDR
    pd_model = fitdist(label(:),'Normal');
    pdf_model = pdf(pd_model,label);
    pd_model_max = max(pdf_model);
    weighting_type = 'addition'; % our P loss
else % no distribution -> no cost C
    pd_model = [];
    pd_model_max = [];
    weighting_type = ''; 
end

opts.pd_model = pd_model; % pd for the chosen model (kernel, normal or nothing[])
opts.pd_model_max = pd_model_max; % maximum value of pdf for the chosen model
opts.weighting_type = weighting_type; % either 'additive weighting factor' or 'multiplied weighting factor'

% --------------------------------------------------------------------
%                                                      Balance data if
%                                                      selected by user
% --------------------------------------------------------------------
if(isequal(opts.method,'o') || isequal(opts.method,'u')|| isequal(opts.method,'s'))
    imdb = balance_data(opts.method, opts, imdb, opts.dataset);
end


%% -------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
%1. setup the batch size
%opts.batchSize is the number of training images in each batch.

if length(imdb.images.labels) > 100000
opts.batchSize = 300 ;
elseif length(imdb.images.labels) > 50000
opts.batchSize = 150;
elseif length(imdb.images.labels) > 20000
    opts.batchSize = 100;
elseif length(imdb.images.labels) > 12000
     opts.batchSize = 50;
elseif length(imdb.images.labels) > 1000
    opts.batchSize = 10;
else
    opts.batchSize = 5;
end

opts.errorFunction = 'euclideanloss';
[net, info] = cnn_train_r_relevance(net, imdb, @getBatch, opts, ...
'val', find(imdb.images.set == 2)) ;

if(isequal(opts.pIndex,'tgm'))
    error_per_epoch = sqrt(info.val.error(1,:) ./ info.val.relevance(1,:) .* info.val.error(2,:) ./ info.val.relevance(2,:));
elseif(isequal(opts.pIndex,'tcwa'))
       w= 2/3;
        error_per_epochgm = sqrt(info.val.error(1,:) ./ info.val.relevance(1,:) .* info.val.error(2,:) ./ info.val.relevance(2,:)); % to be deleted later
        error_per_epoch = w .* info.val.error(1,:) ./ info.val.relevance(1,:) + (1-w) .* info.val.error(2,:) ./ info.val.relevance(2,:);
else
    error_per_epoch =info.val.error(1,:);
end
[minmgm,min_indgm] = min(error_per_epochgm); %to be deleted
[minm,min_ind] = min(error_per_epoch);

fprintf('Lowest gme error is %f %d\n',minmgm .*s_factor, min_indgm) %to be deleted
fprintf('Lowest %s error is %f %d\n',opts.pIndex,minm .*s_factor, min_ind)
    
end

%% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
%getBatch is called by cnn_train.

%'imdb' is the image database.
%'batch' is the indices of the images chosen for this batch.

%'im' is the height x width x channels x num_images stack of images. 
%'labels' indicates the ground truth category of each image.
%N = length(batch);
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

end


