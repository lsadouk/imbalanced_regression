%function  g_mean
%% compute the G-mean of the REC curve

%%1. Compute rec_curve TNR
tE=0; tM=0.7;
d_sort_TNR = rec_curve_extremes(tE, tM);
%%2. Compute rec_curve TPR
tE=0.7; tM=1;
d_sort_TPR = rec_curve_extremes(tE, tM);
%%3. Compute g_mean

g_mean_sort = sqrt(d_sort_TNR .* d_sort_TPR);

figure,
hold on
for i=1:3 % was 2
    if i==1, lineSpec='b-'; elseif i==2, lineSpec='r-'; else, lineSpec='c-'; end
    %plot the REC curve with specified line format
    plot([0;g_mean_sort(:,1,i)],[0;g_mean_sort(:,2,i)],lineSpec);
    
end