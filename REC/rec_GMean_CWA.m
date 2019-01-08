%function  rec_GMean_CWA

% Goal: displays TPR REC curve, TNR REC curve, and GMean or CWA REC curve for the 4 methods that 
% are within the following list:    
%       REC_list = {'L0_n','L0_u','L0_o','L2_n'} where
%       'L0_n': L2_Loss w/ Unbalanced dataset
%       'L0_u': L2_Loss w/ Balanced dataset (using undersampling method)
%       'L0_o': L2_Loss w/ Balanced dataset (using oversampling method)
%       'L2_n': Probabilistic_Loss (kernel method) w/ Unbalanced dataset
% Steps:
%   1. compute the TNR REC curve points for each method
%   2. compute the TPR REC curve points for each method
%   3.1 do interpolation to get interpolated TNR REC curve points
%   3.2 do interpolation to get interpolated TNR REC curve points
%   3.3 compute the g-mean or CWA REC curve points for each method, 3.4 plot them
%   and 3.5 compute the area under the CDF curve for each method

%%global variables
exp_folder_fig = fullfile('result_REC_plots','_fig'); % folder where REC plots are saved in .fig format
exp_folder_png = fullfile('result_REC_plots','_png'); % folder where REC plots are saved in .png format
if ~exist(exp_folder_fig, 'dir'), mkdir(exp_folder_fig) ; end
if ~exist(exp_folder_png, 'dir'), mkdir(exp_folder_png) ; end


%% 0. inputs
REC_list = {'L0_n','L0_u','L0_o','L2_n'};
REC_metric =input('Please select one REC metric: (GMean)/(CWA) ', 's');
if isequal(REC_metric, 'CWA'), w =input('Please select the weight w for CWA from 0 to 1: '); end
dataset =input('Please select one of the following datasets: (abalone)/(accel)/(heat)/(cpuSm)/(bank8FM)/(parkinson)/(dAiler) ', 's');
%% 1. Compute rec_curve TNR
fprintf('(1)Computing AOCs of the TNR RECs: \n');
tE=0; tM=0.7;
[d_sort_TNR s_factor]= rec_curve_extremes_unnormalized_data(tE, tM, dataset,REC_list);
%filename = fullfile(exp_folder, strcat(dataset,'_REC_TNR'));    
print(fullfile(exp_folder_png,strcat(dataset,'_REC_TNR')) ,'-dpng','-r300') %save as png file print TNR REC curve w/ 300 resolution
savefig(fullfile(exp_folder_fig,strcat(dataset,'_REC_TNR','.fig')));% save as .fig file

%% 2. Compute rec_curve TPR
fprintf('(2)Computing AOCs of the TPR RECs: \n');
tE=0.7; tM=1;
[d_sort_TPR s_factor] = rec_curve_extremes_unnormalized_data(tE, tM, dataset,REC_list);
%filename = fullfile(exp_folder, strcat(dataset,'_REC_TPR'));    
print(fullfile(exp_folder_png,strcat(dataset,'_REC_TPR')) ,'-dpng','-r300') %save as png file print TNR REC curve w/ 300 resolution
savefig(fullfile(exp_folder_fig,strcat(dataset,'_REC_TPR','.fig')));% save as .fig file

%% 3. Compute g_mean
fprintf('(3)Computing AOCs of %s RECs for (1)l_2 Unb.,(2)l_2 Bal_u,(3)l_2 Bal_o,(4)l_P Unb.: \n', REC_metric);
figure,
hold on
nb_methods = length(REC_list);
max_error = max(max(d_sort_TPR(:,1,:)));
Ty = 0:0.001:max_error;
metric_sort = zeros(length(Ty),2,nb_methods);

%plot parameters
axis([0 0.28*s_factor 0.0 1.0]); %xlim([0 0.28*70]) % was 0.35 
xlabel(['tolerance ', char(1013)]);
ylabel('Accuracy');


for method=1:nb_methods % was 2
    %% 3.1. do linear interpolation on TPR values to get interpolated values at specific query points 
    irregTx = d_sort_TPR(:,1,method); % tolerance (x-axis)
    x = d_sort_TPR(:,2,method); % accuracy (y-axis)    
    %get only unique values of x and irregTx in order to use interpolation
    %(can't use interpolation when irregTx (x-axis) has duplicates
    [unique_irregTx, unique_index, ~] = unique(irregTx);
    unique_x = x(unique_index);
    
    % OPTION 2 % [y, Ty] = resample(double(x), double(irregTx) ,1/0.001,'spline'); plot(irregTx,x,'.-', Ty,y,'o-')
    y_TPR = interp1(unique_irregTx, unique_x, Ty,'nearest', 'extrap'); %'linear'
    y_TPR(y_TPR<0) = 0;  y_TPR(y_TPR>1) = 1;
    %plot(irregTx,x,'.-', Ty,y_TPR,'o-');
    
    %% 3.2. do linear interpolation on TNR values to get interpolated values at specific query points 
    x = d_sort_TNR(:,2,method);
    irregTx = d_sort_TNR(:,1,method);
    %get only unique values of x and irregTx in order to use interpolation
    %(can't use interpolation when irregTx (x-axis) has duplicates
    [unique_irregTx, unique_index, ~] = unique(irregTx);
    unique_x = x(unique_index);

    y_TNR = interp1(unique_irregTx, unique_x, Ty,'nearest', 'extrap'); 
    y_TNR(y_TNR<0) = 0;  y_TNR(y_TNR>1) = 1;
    %plot(irregTx,x,'.-', Ty,y_TNR,'o-');
    
    %% 3.3. compute g_mean
    metric_sort(:,1,method) = Ty;
    if isequal(REC_metric, 'GMean'),  metric_sort(:,2,method) = sqrt(y_TPR .* y_TNR); REC_title='REC_G_-_M_e_a_n';
    else, metric_sort(:,2,method) = w .* y_TPR + (1-w) .* y_TNR; REC_title='REC_C_W_A';
    end
    if method==1, lineSpec='b-'; elseif method==2, lineSpec='g-'; elseif method==3, lineSpec='c-'; else, lineSpec='r-'; end
    
    %% 3.4. plot the REC curve with specified line format
    plot([0;metric_sort(:,1,method)],[0;metric_sort(:,2,method)],lineSpec); % absice are errors/diff and coordonates are g_mean accuracy per error/diff

    %% 3.5. use Trapezoidal Rule to compute the area under the CDF curve
    n = size(metric_sort,1); % OR n = length(Ty);
    dif_x = [0; metric_sort(1:n-1,1,method)]; 
    dif_xp = [0; metric_sort(1:n-1,2,method)];
    area_under = sum( ((metric_sort(:,2,method)+dif_xp)/2) .* (metric_sort(:,1,method)-dif_x) );

    %% 9. compute the area over the curve by subtracting area under the 
    %curve from the full area
    area_over = metric_sort(n,1,method) * metric_sort(n,2,method) - area_under;
    fprintf('AOC of %d = %f\n',method, area_over);
    
    %% 10. print GMean-CWA REC curves w/ 300 resolution
    %filename = strcat(dataset,'_REC_TPR.png');    print('-f2',filename,'-dpng','-r300') %print TPR REC curve w/ 300 resolution

end

% other figure properties
legend('l_2 Unb.','l_2 Bal_u','l_2 Bal_o' ,'l_P Unb.', 'Location', 'east') %Bal.(o1) L2(L=0) Loss %Unb. SVM
title(REC_title);%title(dataset);
hold off

% save REC G-mean/CWA plot
print(fullfile(exp_folder_png,strcat(dataset,'_REC_', REC_metric)) ,'-dpng','-r300') %save as png file print TNR REC curve w/ 300 resolution
savefig(fullfile(exp_folder_fig,strcat(dataset,'_REC_', REC_metric,'.fig')));% save as .fig file

%print(filename,'-dpng','-r300') %print  REC curve w/ 300 resolution
%savefig(strcat(filename,'.fig'));
