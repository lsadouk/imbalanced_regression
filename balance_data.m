function imdb = balance_data(method, opts, imdb, dataset)
% This function is called by undersampling (u) or oversampling (o) methods
% are called by the user
% Goal: balance the training dataset for regression i.e., undersample or
%  oversample if selected by user, then return the balanced data
% inputs of this function are:
    % method is the method selected (undersampling, oversampling)
    % opts.pd_model_pmeasure if selected method is under, or over-sampling
    % opts.pd_model_max_pmeasure if selected method is under, or over-sampling
    % opts.kfold for k-fold cross validation
    % dataset is the name of the dataset selected by the user
% outputs is imdb, a struct containing input X and output Y
%% 4. 

    
index = find(imdb.images.set ==  1); % take only testing data
labels = imdb.images.labels(1,index); %label = label(:);
data = imdb.images.data(:,:,:, index); %label = label(:);
% 4.1 find relevance of p0    
relevance_labels = 1- pdf(opts.pd_model_pmeasure,labels) ./ opts.pd_model_max_pmeasure;
tE = 0.7; % relevance threshold
ind_labels_supTe = find(relevance_labels >= tE);
ind_labels_infTe =find(relevance_labels < tE); %find(1:length(labels) ~= ind_labels_supTe);
    
if isequal(method, 'u') % do undersampling    
    sampling_rate =input('Please select the undersampling rate 0.5, 1(default), 2: ');
    nb_rare = length(ind_labels_supTe);
    nb_samples = round((1+sampling_rate) * nb_rare);
    nb_frequent = nb_samples-nb_rare;
else % 'o', do oversampling
    sampling_rate =input('Please select the oversampling rate 0.5, 1(default), 2: ');
    nb_frequent = length(ind_labels_infTe);
    nb_samples = round((1+sampling_rate) * nb_frequent);
    nb_rare = nb_samples-nb_frequent;
end

balanced_data = zeros(size(data,1),size(data,2), size(data,3), nb_samples);
balanced_labels = zeros(size(labels,1), nb_samples);
    
if isequal(method, 'u') % do undersampling    
    % rare
    balanced_data(:,:,:,1:nb_rare) = data(:,:,:,ind_labels_supTe);
    balanced_labels(:,1:nb_rare) = labels(:, ind_labels_supTe);
    % frequent
    ind_labels_infTe = randsample(ind_labels_infTe,nb_frequent);
    balanced_data(:,:,:,nb_rare+1:end) = data(:,:,:,ind_labels_infTe); % ind_labels_infTe(1:nb_frequent)
    balanced_labels(:,nb_rare+1:end) = labels(:, ind_labels_infTe); % ind_labels_infTe(1:nb_frequent)
else % 'o', do oversampling
    % frequent
    balanced_data(:,:,:,1:nb_frequent) = data(:,:,:,ind_labels_infTe);
    balanced_labels(:,1:nb_frequent) = labels(:, ind_labels_infTe);
    % rare
    % oversample rare instances
    ind_labels_supTe = datasample(ind_labels_supTe, nb_rare);
    balanced_data(:,:,:,nb_frequent+1:end) = data(:,:,:,ind_labels_supTe); 
    balanced_labels(:,nb_frequent+1:end) = labels(:, ind_labels_supTe);
end
% shuffle training dataset
randNdx=randperm(nb_samples);
balanced_data = balanced_data(:,:,:,randNdx); % 10*20*3*177120
balanced_labels = balanced_labels(:, randNdx); % 1*177120


% assign testing data and labels into imdb
test_indices = find(imdb.images.set == 2);
imdb.images.set = imdb.images.set(test_indices);
imdb.images.data = imdb.images.data(:,:,:, test_indices);
imdb.images.labels = imdb.images.labels(:, test_indices);

% assign balanced training data and label to imdb
imdb.images.set(end+1:end+nb_samples) = 1;
imdb.images.data(:,:,:,end+1:end+nb_samples) = balanced_data;
imdb.images.labels(:,end+1:end+nb_samples) = balanced_labels;

imdb_filename = fullfile('data_preprocessed', strcat('imdb_',dataset,'_r','_balanced_', method, int2str(sampling_rate) ,'.mat'));
save(imdb_filename ,'imdb');
end