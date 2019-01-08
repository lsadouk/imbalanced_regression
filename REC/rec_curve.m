%The function AOC = rec_curve(error_metric,y,yhat,lineSpec) is used
%to draw an REC curve based on the residual y-yhat information and 
%return the area over the REC curve. Note this REC plot is scaled 
%by the mean model, i.e., the mean of the actual response.
%Author: Jinbo Bi (bij2@rpi.edu) 2/25/2003
%Inputs: error_metric -- the type of the error metric, if it is 
%        'AD', the REC curve is based on absolute deviation; if 
%        it is 'SE', the REC curve is based on squared residual.
%        y -- the actual values of response.
%        yhat -- the predicted values of response.
%        lineSpec -- the line specification of the REC curve, for
%        example, if it is 'r-', the REC curve will be a red color
%        solid line, please see MatLab line specification syntax
%        for detail.
%Outputs: AOC -- the area over the REC curve.
%        x_sort -- a matrix of two columns, and the first column
%        is orginal x sorted in ascending order and the second
%        column is the probability of x.
%function [d_sort,AOC]= rec_curve(error_metric,y,yhat,lineSpec)

function [d_sort,AOC] = rec_curve(y,yhat,lineSpec,s_factor)
%y = result(:,2);
%yhat = result(:,1);
%lineSpec = 'r-';
error_metric = 'AD';

%calculate the sample errors and size of the figure box
if error_metric == 'AD'
    %compute the absolute residual
    diff = abs(y- yhat);
    %compute the size of the figure box
    %d = abs(y-mean(y)); % real one
    d= 0.28 * s_factor; % was 0.35 when error range is [0,1] %%%%%%CHANGED  for code rec_curve_extremes_unnormalized_data.m
    %d = abs(y-mean(yhat));
    axis([0 max(d) 0.0 1.0]);
    xxlabel = 'Absolute deviation';
end
if error_metric == 'SE'
    %compute the squared residual
    diff = (y- yhat).*(y-yhat);
    %comptue the size of the box
    d = (y-mean(y)).*(y-mean(y));
    axis([0 max(d) 0.0 1.0]);
    xxlabel = 'Squared residual';
end

%esitmate the cumulative distribution function
[d_sort,AOC,AUC] = CDF(diff);
%d_sort = [ 0 0; d_sort]; plot(d_sort(:,1),d_sort(:,2),lineSpec);
%plot the REC curve with specified line format
plot([0;d_sort(:,1)],[0;d_sort(:,2)],lineSpec);

%xlabel(xxlabel,'fontsize',12); % TO BE PUT LATER
%ylabel('Accuracy','fontsize',12); % TO BE PUT LATER
