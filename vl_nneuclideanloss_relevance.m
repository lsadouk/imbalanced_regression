
function [ Y , benchmark_diff ] = vl_nneuclideanloss_relevance(X, c, pd_model, pd_model_max, weighting_type, dzdy)

assert(numel(X) == numel(c));
%assert(all(size(X) == size(c)));
  
if isempty(pd_model) % lambda = 0
    relevance = 0;
else % lambda =1 or 2
    relevance = 1- pdf(pd_model,c) ./ pd_model_max;
end
  
c= reshape(c,1,1,1,[]);
relevance= reshape(relevance,1,1,1,[]);
assert(all(size(X) == size(c)));

if nargin == 5 || (nargin == 6 && isempty(dzdy))
    %Y = 1 / 2 * sum((X - c).^ 2); % original version
    
    if(isequal(weighting_type,'addition'))
        benchmark_diff = (X - c).^ 2 + abs(X - c) .* relevance ; % BEST weighted version  Probabilistic loss function lp
    elseif(isequal(weighting_type,'multiplication'))
        benchmark_diff = (X - c).^ 2 .* relevance ; % weighted version by multiplying weights
    else %if(isequal(weighting_type,'')) % for L2 loss
        benchmark_diff = (X - c).^ 2;
    end
    Y = 1 / 2 * sum( benchmark_diff ); % .* 2-pdf
     
elseif nargin == 6 && ~isempty(dzdy)
    assert(numel(dzdy) == 1);
       
    Xmc = X-c;
    Xmc(Xmc < 0) = -1;
    Xmc(Xmc >= 0) = 1;
    benchmark_diff = [];
    if(isequal(weighting_type,'addition'))
        Y = bsxfun(@times,dzdy , (X - c) + 0.5 .* Xmc .* relevance  );%%partial derivative of Probabilistic loss function lp
    elseif(isequal(weighting_type,'multiplication'))
        Y = bsxfun(@times,dzdy , (X - c) .* relevance  ); %partial derivative of multiplying weights
    else %if(isequal(weighting_type,'')) % for L2 loss
        Y = bsxfun(@times,dzdy ,(X - c)); % original partial derivative
    end
    
end
