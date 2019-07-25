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

    
index = find(imdb.images.set ==  1); % take only training data
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
elseif isequal(method, 'o') % 'o', do oversampling
    sampling_rate =input('Please select the oversampling rate 0.5, 1(default), 2: ');
    nb_frequent = length(ind_labels_infTe);
    nb_samples = round((1+sampling_rate) * nb_frequent);
    nb_rare = nb_samples-nb_frequent;
else %if isequal(method, 'Smoter') % 'Smoter o2_u0.5', do synthetic oversampling rare + undersampling frequent
    % o2_u0.5 => new_nb_rare = nb_rare*3 , new_nb_frequent = new_nb_rare/2
    sampling_rate = 1; %i.e 200% oversampling and 50% undersampling
    nb_rare_original = length(ind_labels_supTe);
    nb_samples = 3 * nb_rare_original + floor(3 * nb_rare_original / 2); % nb_samples = nb_rare_new + nb_frequent_new
    
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
elseif isequal(method, 'o') % 'o', do oversampling
    % frequent
    balanced_data(:,:,:,1:nb_frequent) = data(:,:,:,ind_labels_infTe);
    balanced_labels(:,1:nb_frequent) = labels(:, ind_labels_infTe);
    % rare
    % oversample rare instances
    ind_labels_supTe = datasample(ind_labels_supTe, nb_rare);
    balanced_data(:,:,:,nb_frequent+1:end) = data(:,:,:,ind_labels_supTe); 
    balanced_labels(:,nb_frequent+1:end) = labels(:, ind_labels_supTe);
else %if isequal(method, 'Smoter') % 'Smoter o2_u0.5', do synthetic oversampling rare + undersampling frequent
    % o2_u0.5 => new_nb_rare = nb_rare*3 , new_nb_frequent = new_nb_rare/2
    
    rare_data = data(:,:,:,ind_labels_supTe);
    rare_data = reshape(rare_data, size(rare_data,1), size(rare_data,4)); % dimension attributes*1*1*nb_samples to attributes*nb_samples
    rare_data = rare_data'; % nb_samples*attibutes (602*8 for abalone)
    rare_labels = labels(1, ind_labels_supTe); % 1*nb_samples (1*602)
    count = 0;
    for i=1:1:nb_rare_original % for each rare instance generate 2 synthetic instances 
        rare_data_excl_i = rare_data(1:end ~= i,:);
        rare_label_excl_i = rare_labels(1, 1:end ~= i);
        nns = knnsearch(rare_data_excl_i,rare_data(i,:),'K',2); % find the 2 nearest neighbor attibrutes of instance i % knnsearch(data where to search, what_look_for,'K', nb_neighbors)
        for j=1:1:size(nns,2) % for each of the 2 new synthetic instance, compte new instance
           % 1. compute new rare data (attributes)
           try
                 nn= nns(j);
           catch ME
              rethrow(ME)
           end
           nn_data = rare_data_excl_i(nn,:);
           nn_label = rare_label_excl_i(1,nn);
           diff = rare_data(i,:) - nn_data;
           count = count+1; % increment count of new synthesized instances
           new_rare_data(count,:) = rare_data(i,:) + diff .* randsample(0:1,size(rare_data,2),true); %randsample=Draw nb_attributes values with replacement from the integers 0:1.
           
           % 1. compute new rare target variable
           dist1 = norm(new_rare_data(count,:) - rare_data(i,:)); % euclidean distance between new and old original data
           dist2 = norm(new_rare_data(count,:) - nn_data); % euclidean distance between new and neighrest neighbor of original data
           rare_label = (rare_labels(1,i) * dist2 + nn_label * dist1) ./ (dist2 + dist1);
           if   isnan(rare_label) == 1
               resultttt = 'error';
           end
           new_rare_label(1,count)= rare_label ;
        end
    end
    % 1. add rare data
    balanced_data(:,:,:,1:nb_rare_original) = data(:,:,:,ind_labels_supTe);
    balanced_labels(:,1:nb_rare_original) = labels(:, ind_labels_supTe);
    % 2. add synthesized rare data
    new_rare_data = reshape(new_rare_data', size(new_rare_data',1),1,1,size(new_rare_data',2));  %convert new_rare_data from nbsamples*attributes back to attributes*1*1*nb_samples
    balanced_data(:,:,:,nb_rare_original+1:nb_rare_original+count) = new_rare_data;
    balanced_labels(:,nb_rare_original+1:nb_rare_original+count) = new_rare_label;
    % 3. add frequent data (with undersampling from
    % length(ind_labels_infTe) to 3*nb_rare_original/2
    ind_labels_infTe = randsample(ind_labels_infTe, floor(3*nb_rare_original/2));
    balanced_data(:,:,:,nb_rare_original+count+1:end) = data(:,:,:,ind_labels_infTe);
    balanced_labels(:,nb_rare_original+count+1:end) = labels(:, ind_labels_infTe);

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

%sometimes, some input data has same attributes => which gives us NAN in
%labels ==> need to delted these entries
indices = find(isnan(imdb.images.labels) == 0);
imdb.images.data = imdb.images.data(:,:,:,indices);
imdb.images.labels = imdb.images.labels(:,indices);
imdb.images.set = imdb.images.set(indices);

imdb_filename = fullfile('data_preprocessed', strcat('imdb_',dataset,'_r','_balanced_', method, int2str(sampling_rate) ,'.mat'));
save(imdb_filename ,'imdb');
end