
function [ Y , benchmark_diff ] = vl_nneuclideanloss_relevance(X, c, pd_model, pd_model_max, dzdy)

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

if nargin == 4 || (nargin == 5 && isempty(dzdy))
    %Y = 1 / 2 * sum((X - c).^ 2); % original version
    benchmark_diff = (X - c).^ 2 + abs(X - c) .* relevance ; %weighted version  Probabilistic loss function lp

    Y = 1 / 2 * sum( benchmark_diff ); % .* 2-pdf
     
elseif nargin == 5 && ~isempty(dzdy)
    assert(numel(dzdy) == 1);
    %Y = bsxfun(@times,dzdy ,(X - c)); % original partial derivative
    
    Xmc = X-c;
    Xmc(Xmc < 0) = -1;
    Xmc(Xmc >= 0) = 1;
    benchmark_diff = [];
    Y = bsxfun(@times,dzdy , (X - c) + 0.5 .* Xmc .* relevance  ); %weighted partial derivative

end
