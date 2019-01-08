%The function [x_sort,area_over,area_under]=CDF(x) is used to 
%estimate the cumulative distribution function of the random 
%variable x and compute the areas under and over the CDF curve.
%Author: Jinbo Bi (bij2@rpi.edu) 2/25/2003
%Inputs: x -- a vector of real numbers as a sample of random
%        variable x.
%Outputs: x_sort -- a matrix of two columns, and the first column
%        is orginal x sorted in ascending order and the second
%        column is the probability of x.
%        area_over -- the area over the CDF curve, a real number.
%        area_under -- the area under the CDF curve.

function [x_sort,area_over,area_under]=CDF(x)

%compute the probability for each x in ascending order
f = sort(x);
m = length(f);
t = f(1);
count = 1;
index = 1;
for i=2:m
	%if f(i) > t % deleted because we want to have the same number of
	%points for each of the REC_Lists: l2_unb, l2_u, l2_o, and lp_unb
    % SO we keep duplicates (which won't affect the REC plot)
		x_sort(index,1) = t;
		x_sort(index,2) = count/m;
		t = f(i);
		index = index+1;
	%end
	count = count+1;
end
x_sort(index,1) = f(m);
x_sort(index,2) = 1;
	
%use Trapezoidal Rule to compute the area under the CDF curve
n = size(x_sort,1);
dif_x = [0; x_sort(1:n-1,1)];
dif_xp = [0; x_sort(1:n-1,2)];
area_under = sum( ((x_sort(:,2)+dif_xp)/2) .* (x_sort(:,1)-dif_x) );

%compute the area over the curve by subtracting area under the 
%curve from the full area
area_over = x_sort(n,1)*x_sort(n,2) - area_under;
