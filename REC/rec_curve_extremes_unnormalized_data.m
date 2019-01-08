function [d_sort s_factor]= rec_curve_extremes_unnormalized_data(tE, tM, dataset,REC_list)

% outputs d_sort is nx2x3 where n is the number of samples in testing set,
% 2 are the columns, the first represents the differences, the second
% represents their corresponding (cumulative) accuracies, and 3 corresponds
% to the three models (L2_loss, prob_loss, undersample_loss)

addpath('../');
if(isempty(dataset))
    dataset =input('Please one of the following datasets: (abalone)/ (YearPredictionMSD)/(availPwr)/(bank8FM)/(cpuSm)/(boston)/(heat)/(accel)/(fuelCons)/(maxTorque)/(parkinson) ', 's');
end
%function rec_curve_extremes(dataset, lambda, pIndex)
%% 1. get data
load(strcat('../data_preprocessed/','imdb_',dataset,'_r.mat'));
index = find(imdb.images.set ==  1); % take only testing data
label = imdb.images.labels(index); %label = label(:);
label = label .* s_factor;

%% 2. From data, get pd_model and max(pdf_model)
pd_model = fitdist(label(:),'kernel');
pdf_model = pdf(pd_model,label);
pd_model_max = max(pdf_model);


%% 3. load result labels and predictions
%%% 3.1 since files result_dataset_r_REClist(i) has different size/length depending on the batch size used to extract results
% % we choose to pick the smallest length for all files result_dataset_r_REClist(i)
% for i=1:length(REC_list)
%     result1 = load(strcat('result_',dataset,'_r_',REC_list{i},'.mat')); %load('result_r_L0n.mat')    
% end
figure,
hold on
for i=1:length(REC_list) % 4
    load(strcat('../result_test_data/','result_',dataset,'_r_',REC_list{i},'.mat')); %load('result_r_L2.mat')
    % extreme_labels = result(index,2);
    labels = result(:,2) .* s_factor; % updated in 11sep18 to produce unnormalized labels instead of normalized ones
    predictions = result(:,1) .* s_factor; % updated in 11sep18 to produce unnormalized predictions instead of normalized ones
    %% 4. select only result labels that are above tE (0.7)
    relevance_labels = 1- pdf(pd_model,labels) ./ pd_model_max;
    %tM = 0.5;
    %tE = 0; % was 0.5
    ind_labels_supTe = find(relevance_labels >= tE & relevance_labels < tM);
    ext_labels = labels(ind_labels_supTe);
    ext_predictions = predictions(ind_labels_supTe);
    %if i==1, lineSpec='b-'; else, lineSpec='r-'; end
    if i==1, lineSpec='b-'; elseif i==2, lineSpec='g-'; elseif i==3, lineSpec='c-'; else, lineSpec='r-'; end
    %if i==1, lineSpec='b-'; elseif i==2, lineSpec='c-'; else, lineSpec='r-'; end
    [d_sort(:,:,i), AOC(i)] = rec_curve(ext_labels,ext_predictions,lineSpec, s_factor);
        % 4 = 3_time.5
end
%legend('Square loss function (L=0)','Prob. loss function (L=K*1)')
legend('l_2 Unb.','l_2 Bal_u','l_2 Bal_o' ,'l_P Unb.', 'Location', 'east') %Bal.(o1) L2(L=0) Loss %Unb. SVM
%
if tE==0,    REC_title = 'REC_T_N_R';  % we are dealing with TNR REC
else,     REC_title = 'REC_T_P_R'; end
xlabel(['tolerance ', char(1013)]);
ylabel('Accuracy');
title(REC_title);%title(dataset);
hold off

fprintf('l_2 Unb. AOC = %f\n', AOC(1));
fprintf('l_2 Bal_u AOC = %f\n', AOC(2));
fprintf('l_2 Bal_o1 AOC = %f\n', AOC(3));
fprintf('l_P Unb. AOC = %f\n', AOC(4));