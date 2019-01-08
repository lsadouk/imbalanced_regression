function [imdb, s_factor] = setup_data(kfold, dataset)

%% output:
% a saved file'.mat' w/ variables: (a) imdb struct w. matrix image and
% normalized labels (in range [0,1])        
% (b) s_factor: the scaling factor max_data - min_data (to recover unnormalized original data)


%% 1. Load scenes/frames/images
ScenesPath = 'data/'; %'data/preprocessed_data/';

% read csv file which contains letters (e.g. nominal values)
dataset_file = fullfile( ScenesPath, strcat(dataset,'.csv'));
data = csvread_with_letters(dataset_file,1,0);

%%% save data label
image = data(:,2:end); % Nx6
label = data(:,1)'; % 1xN

%% reshape data from Nx6 (N=#samples, 6=#columns/features) to 6x1x1xN
nb_samples = size(image,1);
nb_features = size(image,2);
image = reshape(image', nb_features,1,1,nb_samples);

%% 2. shuffle the dataset
randNdx=randperm(length(label));
image=image(:,:,:,randNdx); % 10*20*3*177120
label=label(1, randNdx); % 1*177120

%% 3. split data into training & testing set
if(kfold == 0) % NO TRAINING PHASE all images are for testing
    trainData=[];
    trainLabel=[];
    testData=image;
    testLabel=label;
else % if(kfold ~= 0)
    %kfold = 4; % 
    sizekmul =size(image,4)-mod(size(image,4),kfold);  % for 10-fold cross validation %177120
    trainData=image(:,:,:,1:sizekmul/kfold*(kfold-1)); %3/4 samples are for training
    trainLabel=label(:,1:sizekmul/kfold*(kfold-1)); %3/4 samples are for training (10*20*3*132840)
    testData=image(:,:,:,sizekmul/kfold*(kfold-1)+1:sizekmul);%1/4 samples are for training %44280
    testLabel=label(:,sizekmul/kfold*(kfold-1)+1:sizekmul);%1/4 samples are for training
     

%% 4. put all data into final dataset 'imdb'
nb_train = length(trainLabel); %or size(trainLabel,2)  132839
nb_test = length(testLabel); %44280
nb_total = nb_train + nb_test; %177119
image_size = [size(testData,1) size(testData,2) size(testData,3)]; 
imdb.images.data   = zeros(image_size(1), image_size(2),image_size(3), nb_total, 'single');
imdb.images.labels = zeros(1, nb_total, 'single'); % 1*n
imdb.images.set    = zeros(1, nb_total, 'uint8');

if(kfold ~= 0) % NO TRAINING PHASE all images are for testing
    imdb.images.data(:,:,:,1:nb_train) = trainData;
    imdb.images.labels(1, 1:nb_train) = single(trainLabel);
    imdb.images.set(1, 1:nb_train) = 1;
end

imdb.images.data(:,:,:,nb_train+1:nb_train+nb_test) = testData;
imdb.images.labels(1, nb_train+1:nb_train+nb_test) = single(testLabel);
imdb.images.set(:, nb_train+1:nb_train+nb_test) = 2;

%% 5. normalize the data inputs X
imdb.images.data = dimensionNormalize(imdb.images.data); % normalize each attribute separately

%% 6. normalize the labels Y (max label = 24)
%label = label ./max(label); % range of label from 0 to 1
s_factor = max(imdb.images.labels)-min(imdb.images.labels);
imdb.images.labels = (imdb.images.labels-min(imdb.images.labels)) ./ s_factor ; % range from 0 to 1

expDir = fullfile('data_preprocessed');
if ~exist(expDir, 'dir'), mkdir(expDir) ; end

imdb_filename = fullfile('data_preprocessed', strcat('imdb_',dataset,'_r.mat'));
save(imdb_filename ,'imdb', 's_factor');

end
