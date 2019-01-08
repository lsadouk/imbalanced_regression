%function  g_mean DOESNT WORK COZ TOO FEW POINTS are plotted (128 instead
%of 572)
%% compute the G-mean of the REC curve
dataset =input('Please one of the following datasets: (abalone)/ (YearPredictionMSD)/(availPwr)/(bank8FM)/(cpuSm)/(boston)/(heat)/(accel)/(fuelCons)/(maxTorque)/(parkinson) ', 's');
%% 1. Compute rec_curve TNR
tE=0; tM=0.7;
d_sort_TNR = rec_curve_extremes(tE, tM, dataset);
%% 2. Compute rec_curve TPR
tE=0.7; tM=1;
d_sort_TPR = rec_curve_extremes(tE, tM, dataset);
%% 3. Compute g_mean


figure,
hold on
nb_methods = 3;
max_error = max(max(d_sort_TPR(:,1,:)));
%Ty = 0:0.001:max_error;
Ty = d_sort_TPR(:,1,method);
g_mean_sort = zeros(length(Ty),2,nb_methods);
for method=1:nb_methods % was 2
%     %% 4. do linear interpolation on TPR values to get interpolated values at specific query points 
%     x = d_sort_TPR(:,2,method);
%     irregTx = d_sort_TPR(:,1,method);
%     % OPTION 2 % [y, Ty] = resample(double(x), double(irregTx) ,1/0.001,'spline'); plot(irregTx,x,'.-', Ty,y,'o-')
%     y_TPR = interp1(irregTx, x, Ty,'nearest', 'extrap'); %'linear'
%     y_TPR(y_TPR<0) = 0;  y_TPR(y_TPR>1) = 1;
%     %plot(irregTx,x,'.-', Ty,y_TPR,'o-');
    
    %% 5. do linear interpolation on TNR values to get interpolated values at specific query points 
    x = d_sort_TNR(:,2,method);
    irregTx = d_sort_TNR(:,1,method);
    
    y_TNR = interp1(irregTx, x, Ty,'nearest', 'extrap'); 
    y_TNR(y_TNR<0) = 0;  y_TNR(y_TNR>1) = 1;
    %plot(irregTx,x,'.-', Ty,y_TNR,'o-');
    
    %% 6. compute g_mean
    g_mean_sort(:,1,method) = Ty;
    y_TPR = d_sort_TPR(:,2,method);
    g_mean_sort(:,2,method) = sqrt(y_TPR .* y_TNR);
    if method==1, lineSpec='b-'; elseif method==2, lineSpec='r-'; else, lineSpec='c-'; end
    
    %% 7. plot the REC curve with specified line format
    plot([0;g_mean_sort(:,1,method)],[0;g_mean_sort(:,2,method)],lineSpec); % absice are errors/diff and coordonates are g_mean accuracy per error/diff
    xlim([0 0.35])
    legend('Unb. L2_Loss','Unb. Prob._Loss','Bal.(u1) L2_Loss') %Bal.(o1) L2(L=0) Loss %Unb. SVM
    title(strcat('G-Mean REC'));
    %% 8. use Trapezoidal Rule to compute the area under the CDF curve
    n = size(g_mean_sort,1); % OR n = length(Ty);
    dif_x = [0; g_mean_sort(1:n-1,1,method)]; 
    dif_xp = [0; g_mean_sort(1:n-1,2,method)];
    area_under = sum( ((g_mean_sort(:,2,method)+dif_xp)/2) .* (g_mean_sort(:,1,method)-dif_x) );

    %% 9. compute the area over the curve by subtracting area under the 
    %curve from the full area
    area_over = g_mean_sort(n,1,method) * g_mean_sort(n,2,method) - area_under;
    fprintf('AOC of %d = %f\n',method, area_over);
end

