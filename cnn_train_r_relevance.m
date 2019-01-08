function [net, info] = cnn_train_r_relevance(net, imdb, getBatch, varargin)
%CNN_TRAIN  An example implementation of SGD for training CNNs
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option). Multi-GPU
%    support is relatively primitive but sufficient to obtain a
%    noticable speedup.

% Copyright (C) 2014-15 Andrea Vedaldi.
% Modified by Lamyaa Sadouk

opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.gpus = [] ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.prefetch = false ;
opts.cudnn = true ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.prediction_type = 'r';
%opts.lambda = 0; % by default L2 loss function (and not probabilistic loss function)
opts.pd_model = [];
opts.pd_model_max = [];
opts.pd_model_pmeasure = [];
opts.pd_model_max_pmeasure = 0;
opts.pIndex = 'm' ; % by default, set to MAE (m)
opts.lambda = 0;
opts.dataset = '';
opts.method = '';
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
info.train.objective = [] ;
if(isequal(opts.pIndex,'tgm') || isequal(opts.pIndex,'tcwa'))
            info.train.error = zeros(2,0);info.val.error = zeros(2,0); %creates an empty matrix 0x2 containing errors [error_positives, error_negatives]
            info.train.relevance = zeros(2,0); info.val.relevance = zeros(2,0);%creates an empty matrix 0x2 containing relevance [relevance_positives, relevance_negatives]
else
    info.train.error = [] ; info.val.error = [] ;
    info.train.relevance = [] ; info.val.relevance = [] ;
end
info.train.topFiveError = [] ;
info.train.speed = [] ;
info.val.objective = [] ;
info.val.topFiveError = [] ;
info.val.speed = [] ;

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
    % Legacy code: will be removed
    if isfield(net.layers{i}, 'filters')
      net.layers{i}.momentum{1} = zeros(size(net.layers{i}.filters), 'single') ;
      net.layers{i}.momentum{2} = zeros(size(net.layers{i}.biases), 'single') ;
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, 2, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = single([1 0]) ;
      end
    end
  end
end

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end



%% setup error calculation function
if isstr(opts.errorFunction)
  switch opts.errorFunction
%     case 'none'
%       opts.errorFunction = @error_none ;
    case 'multiclass'
      %opts.errorFunction = @error_multiclass ;
      if isempty(opts.errorLabels), opts.errorLabels = {'top1e', 'top5e'} ; end
    case 'binary'
      %opts.errorFunction = @error_binary ;
      if isempty(opts.errorLabels), opts.errorLabels = {'bine'} ; end
    case 'euclideanloss'
        if isempty(opts.errorLabels), opts.errorLabels = {'euclidean'} ; end
 
    otherwise
      error('Uknown error function ''%s''', opts.errorFunction) ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('resuming by loading epoch %d\n', start) ;
  load(modelPath(start), 'net', 'info') ;
end

for epoch=start+1:opts.numEpochs
  info.train.objective(end+1) = 0 ;
  info.train.error(:,end+1) = 0 ;  % was info.train.error(end+1) = 0 ;
  info.train.relevance(:,end+1) = 0 ; % was info.train.relevance(end+1) = 0 ;
  info.train.topFiveError(end+1) = 0 ;
  info.train.speed(end+1) = 0 ;
  info.val.objective(end+1) = 0 ;
  info.val.error(:,end+1) = 0 ;
  info.val.relevance(:,end+1) = 0 ;
  info.val.topFiveError(end+1) = 0 ;
  info.val.speed(end+1) = 0 ;
    
    %% TO BE REPLACED LATER for 1 or 3 input channels
  figure(2) ; clf ; colormap gray ;
  vl_imarraysc(squeeze(net.layers{1}.weights{1}),'spacing',2)
  axis equal ; title('filters in the first layer') ;

  % train one epoch and validate
  %learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  %%above is original code for learningRate
    %learningRate = epoch * opts.learningRate /5;%updated 23-06-2016
  %learningRate = 75*opts.learningRate ./ max(epoch,75);%updated 24-06-2016
  learningRate = opts.learningRate;
  train = opts.train(randperm(numel(opts.train))) ; % shuffle
  val = opts.val ;
  if numGpus <= 1
    [net,info] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net,info) ;
    [~,info] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net,info) ;
  else
    spmd(numGpus)
      [net_, info] = process_epoch(opts, getBatch, epoch, train, learningRate, imdb, net,info) ;
      [~, info] = process_epoch(opts, getBatch, epoch, val, 0, imdb, net_,info) ;
    end
    net = net_{1} ;
%     stats.train = sum([stats_train_{:}],2) ;
%     stats.val = sum([stats_val_{:}],2) ;
  end

  % save
 if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
%   for f = sets
%     f = char(f) ;
%     n = numel(eval(f)) ;
%     info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus) ;
%     info.(f).objective(epoch) = stats.(f)(2)  ; %/ n
%     info.(f).error(:,epoch) = stats.(f)(3:end) ; %/ n
%   end
  if(mod(epoch, 5)==0)
    if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end
  end

  figure(1) ; clf ;
  %hasError = isa(opts.errorFunction, 'function_handle') ;
  hasError = 1;
  subplot(1,1+hasError,1) ;
  if ~evaluateMode
    semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
    hold on ;
  end
  semilogy(1:epoch, info.val.objective, '.--') ;
  xlabel('training epoch') ; ylabel('energy') ;
  grid on ;
  h=legend(sets) ;
  set(h,'color','none');
  title('objective') ;
  
  
 if(isequal(opts.pIndex,'tgm'))
       train_error_per_epoch = sqrt(info.train.error(1,:) ./ info.train.relevance(1,:) .* info.train.error(2,:) ./ info.train.relevance(2,:));
       test_error_per_epoch = sqrt(info.val.error(1,:) ./ info.val.relevance(1,:) .* info.val.error(2,:) ./ info.val.relevance(2,:));
elseif(isequal(opts.pIndex,'tcwa'))
        w= 2/3;
       train_error_per_epoch = w .* info.train.error(1,:) ./ info.train.relevance(1,:) + (1-w).* info.train.error(2,:) ./ info.train.relevance(2,:);
       test_error_per_epoch = w .* info.val.error(1,:) ./ info.val.relevance(1,:) + (1-w) .* info.val.error(2,:) ./ info.val.relevance(2,:);
else
       train_error_per_epoch =info.train.error(1,:);test_error_per_epoch =info.val.error(1,:);
 end
  
  if hasError
    subplot(1,2,2) ; leg = {} ;
    if ~evaluateMode
      plot(1:epoch, train_error_per_epoch', '.-', 'linewidth', 2) ; % was info.train.error
      hold on ;
      leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
    end
    plot(1:epoch, test_error_per_epoch', '.--') ; % was info.val.error
    leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
    set(legend(leg{:}),'color','none') ;
    grid on ;
    xlabel('training epoch') ; ylabel('error') ;
    title('error') ;
  end
  drawnow ;
  print(1, modelFigPath, '-dpdf') ;
end

% % -------------------------------------------------------------------------
% function err = error_multiclass(opts, labels, res)
% % -------------------------------------------------------------------------
% predictions = gather(res(end-1).x) ;
% [~,predictions] = sort(predictions, 3, 'descend') ;
% 
% % be resilient to badly formatted labels
% if numel(labels) == size(predictions, 4)
%   labels = reshape(labels,1,1,1,[]) ;
% end
% 
% % skip null labels
% mass = single(labels(:,:,1,:) > 0) ;
% if size(labels,3) == 2
%   % if there is a second channel in labels, used it as weights
%   mass = mass .* labels(:,:,2,:) ;
%   labels(:,:,2,:) = [] ;
% end
% 
% error = ~bsxfun(@eq, predictions, labels) ;
% err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
% err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:5,:),[],3)))) ;
% 
% % -------------------------------------------------------------------------
% function err = error_binaryclass(opts, labels, res)
% % -------------------------------------------------------------------------
% predictions = gather(res(end-1).x) ;
% error = bsxfun(@times, predictions, labels) < 0 ;
% err = sum(error(:)) ;
% 
% % -------------------------------------------------------------------------
% function err = error_none(opts, labels, res)
% % -------------------------------------------------------------------------
% err = zeros(0,1) ;

% -------------------------------------------------------------------------
function  [net_cpu,info,prof] = process_epoch(opts, getBatch, epoch, subset, learningRate, imdb, net_cpu,info)
% -------------------------------------------------------------------------

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net = vl_simplenn_move(net_cpu, 'gpu') ;
else
  net = net_cpu ;
  net_cpu = [] ;
end

% validation mode if learning rate is zero
%training = learningRate > 0 ;
training = ~(isempty(opts.train) || learningRate == 0);
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 2, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end
res = [] ;
mmap = [] ;
stats = [] ;
start = tic ;
n= 0; % n = #total instances (#t + batchsize) (see below)
for t=1:opts.batchSize:numel(subset)
  fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
  numDone = 0 ;
  error = [] ;
  for s=1:opts.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
    [im, labels] = getBatch(imdb, batch) ;

    if opts.prefetch
      if s==opts.numSubBatches
        batchStart = t + (labindex-1) + opts.batchSize ;
        batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
      getBatch(imdb, nextBatch) ;
    end

    if numGpus >= 1
      im = gpuArray(im) ;
    end

    % evaluate CNN
    net.layers{end}.class = labels ;
    if training, dzdy = one; else, dzdy = [] ; end
    res = vl_simplenn_relevance(net, im, dzdy, res, ...
                      'accumulate', s ~= 1, ...
                      'disableDropout', ~training, ...
                      'conserveMemory', opts.conserveMemory, ...
                      'backPropDepth', opts.backPropDepth, ...
                      'sync', opts.sync, ...
                      'cudnn', opts.cudnn, ...
                      'pd_model', opts.pd_model, ...
                      'pd_model_max', opts.pd_model_max) ; % 'lambda', opts.lambda

%     % accumulate training errors
%     error = sum([error, [...
%       sum(double(gather(res(end).x))) ;
%       reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
%     numDone = numDone + numel(batch) ;
% print information
    %%batch_time = toc(batch_time) ;
    %%speed = numel(batch)/batch_time ;
    if training
        info.train = updateError(opts, info.train, net, res) ;
        infos = info.train;
    else
        info.val = updateError(opts, info.val, net, res) ;
        infos = info.val;
    end
    %fprintf(' %.2f s (%.1f images/s)', batch_time, speed) ;
    n = t + numel(batch) - 1 ;
    
    switch opts.errorFunction
      case 'multiclass'
        fprintf(' err %.1f err5 %.1f', ...
        info.train.error(end)/n*100, infos.topFiveError(end)/n*100) ;
        fprintf('\n') ;
      case 'binary'
        fprintf(' err %.1f', ...
        infos.error(end)/n*100) ;
        fprintf('\n') ;
      case 'euclideanloss'
        %fprintf(' err %.3f', sqrt(infos.error(end) / n));
        if(isequal(opts.pIndex,'g'))
            fprintf(' err %.3f', (infos.error(end)).^(1/n) ); %* 28      
        elseif(isequal(opts.pIndex,'mae')|| isequal(opts.pIndex,'rmse') )
            fprintf(' err %.3f', sqrt(infos.error(end) / n) .* 169.74);
        elseif(isequal(opts.pIndex,'tgm'))
            fprintf(' err %.3f', sqrt(infos.error(1,end) / infos.relevance(1,end) * infos.error(2,end) / infos.relevance(2,end))); % g-mean = sqrt(positive_errors/nb_pos_errors*negative_errors/nb_neg_errors)
        elseif(isequal(opts.pIndex,'tcwa'))
            w= 2/3;
            fprintf(' err %.3f', w .* infos.error(1,end) ./ infos.relevance(1,end) + (1-w) .* infos.error(2,end) ./ infos.relevance(2,end) ); 
        else % if(isequal(opts.pIndex,'w') || isequal(opts.pIndex,'wm') || isequal(opts.pIndex,'swm') 
            fprintf(' err %.3f', infos.error(end) / infos.relevance(end)); % * 28 delete 28 later on
        end
        fprintf('\n') ;
    end
  end
    
  % gather and accumulate gradients across labs
  if training
    if numGpus <= 1
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
    end
  end

  % print learning statistics

  time = toc(start) ;
  %stats = sum([stats,[0 ; error]],2); % works even when stats=[]
%   stats(1) = time ;
%   stats(2) = info.train.objective(end);
%   stats(3) = info.train.error(end);
  nn = (t + batchSize - 1) / max(1,numlabs) ;
  speed = nn/time ;
  fprintf('%.1f Hz%s\n', speed) ;

%   fprintf(' obj:%.3g', stats(2)/n) ;
%   for i=1:numel(opts.errorLabels)
%     fprintf(' %s:%.3g', opts.errorLabels{i}, stats(i+2)/n) ;
%   end
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end
  %code added to save the predictions =res.x(end-1) as well as labels
  pred = gather(res(end-1).x) ;
  %[~,pred] = max(pred,[], 3) ; %for classification (multiple classes)
  if(length(pred(:)) == opts.batchSize)
        result(t:t+opts.batchSize-1,1) = pred(:);
        result(t:t+opts.batchSize-1,2) = labels(:);
  end
end


if(n ~=0) % if we have subsets (i.e. training or val instances)
    expResultsDir = fullfile('result_test_data');
    if ~exist(expResultsDir, 'dir'), mkdir(expResultsDir) ; end

    result_file = strcat('result_test_data/result_',opts.dataset,'_r_L',int2str(opts.lambda),'_', opts.method, '.mat');
    save(result_file, 'result');
    if(isequal(opts.pIndex,'mae'))
       if training, info.train.error(end) = info.train.error(end) / n; 
        else, info.val.error(end) = info.val.error(end) / n; end
    elseif(isequal(opts.pIndex,'rmse'))
        if training, info.train.error(end) = sqrt(info.train.error(end) / n); 
        else, info.val.error(end) = sqrt(info.val.error(end) / n); end
    end
%     if training,   infos = info.train; else,     infos = info.val;  end
%         %info.train.error(end) = sqrt(info.train.error(end) / n);
%     if(isequal(opts.pIndex,'g'))
%         infos.error(end) = infos.error(end).^(1/n) ;
% %        infos.error(end) = infos.error(end) ./ infos.relevance(end) ; % info.relevance=number of instances considered
%     elseif(isequal(opts.pIndex,'m'))
%         infos.error(end) = infos.error(end) ./ n  .* 28; % delete 28 later on
%     elseif(isequal(opts.pIndex,'tgm'))
%             fprintf(' err %.3f', sqrt(infos.error(1,end) / infos.relevance(1,end) * infos.error(2,end) / infos.relevance(2,end))); % g-mean = sqrt(positive_errors/nb_pos_errors*negative_errors/nb_neg_errors)
%     else %if(isequal(opts.pIndex,'w') || isequal(opts.pIndex,'wm') || isequal(opts.pIndex,'swm') || isequal(opts.pIndex,'tgm'))
%         infos.error(end) = infos.error(end) ./  infos.relevance(end) ; % .* 28 delete 28 later on
%     end
%     if training,   info.train = infos; else,     info.val = infos ;  end
end
if nargout > 2
  prof = mpiprofile('info');
  mpiprofile off ;
end

if numGpus >= 1
  net_cpu = vl_simplenn_move(net, 'cpu') ;
else
  net_cpu = net ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l=numel(net.layers):-1:1
  for j=1:numel(res(l).dzdw)
    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
    thisLR = lr * net.layers{l}.learningRate(j) ;
    
%     net.layers{l}
%     net.layers{l}.weights
%     res(l).dzdw
%     pause

    % accumualte from multiple labs (GPUs) if needed
    if nargin >= 6
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end

    if isfield(net.layers{l}, 'weights')
      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + thisLR * net.layers{l}.momentum{j} ;
    else
      % Legacy code: to be removed
      if j == 1
        net.layers{l}.momentum{j} = ...
          opts.momentum * net.layers{l}.momentum{j} ...
          - thisDecay * net.layers{l}.filters ...
          - (1 / batchSize) * res(l).dzdw{j} ;
        net.layers{l}.filters = net.layers{l}.filters + thisLR * net.layers{l}.momentum{j} ;
      else
        net.layers{l}.momentum{j} = ...
          opts.momentum * net.layers{l}.momentum{j} ...
          - thisDecay * net.layers{l}.biases ...
          - (1 / batchSize) * res(l).dzdw{j} ;
        net.layers{l}.biases = net.layers{l}.biases + thisLR * net.layers{l}.momentum{j} ;
      end
    end
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function info = updateError(opts, info, net, res)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

labels = net.layers{end}.class ;
labels= reshape(labels,1,1,1,[]);
% info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ; % old version
info.objective(end) = info.objective(end) + sum((predictions - labels).^ 2) ;
    
%info.speed(end) = info.speed(end) + speed ;
switch opts.errorFunction
  case 'multiclass'
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    info.error(end) = info.error(end) +....
      sum(sum(sum(error(:,:,1,:))))/n ;
    info.topFiveError(end) = info.topFiveError(end) + ...
      sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
  case 'binary'
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error(end) = info.error(end) + sum(error(:))/n ;
  case 'euclideanloss'
    % error = euclideanloss(predictions, labels); %for unweighted euclidean loss
    
    if(isequal(opts.pIndex,'w'))
        %% WMAE
        relevance_labels = 1- pdf(opts.pd_model_pmeasure,labels) ./ opts.pd_model_max_pmeasure;
        error =   nansum(relevance_labels .* abs(predictions - labels) ); %  sum((predictions - labels).^ 2);
        info.relevance(end) = info.relevance(end) + nansum(relevance_labels);
    elseif(isequal(opts.pIndex,'t'))
        %% TMAE
        diff =   abs(predictions - labels) ; %  sum((predictions - labels).^ 2);
        relevance_labels = 1- pdf(opts.pd_model_pmeasure,labels) ./ opts.pd_model_max_pmeasure;
        ind_labels_supTe = find(relevance_labels >= 0.7);
        error = nansum(diff(ind_labels_supTe));
        info.relevance(end) = info.relevance(end) + length(relevance_labels(ind_labels_supTe)); % number of instances considered (having relevance > tE)
    elseif(isequal(opts.pIndex,'tgm') || isequal(opts.pIndex,'tcwa'))  %
        %% Threshold g-mean
        diff =   abs(predictions - labels) ; %  sum((predictions - labels).^ 2);
        relevance_labels = 1- pdf(opts.pd_model_pmeasure,labels) ./ opts.pd_model_max_pmeasure;
        ind_labels_supTe = find(relevance_labels >= 0.7);
        ind_labels_infTe = find(relevance_labels < 0.7);
        %error = nansum(diff(ind_labels_supTe) ./ labels(ind_labels_supTe));
        %info.relevance(end) = info.relevance(end) + length(relevance_labels(ind_labels_supTe)); % number of instances considered (having relevance > tE)
        error = [nansum(diff(ind_labels_supTe)) ; nansum(diff(ind_labels_infTe))];
        info.relevance(:,end) = info.relevance(:,end) + [length(ind_labels_supTe) ; length(ind_labels_infTe)]; % number of instances considered (having relevance > tE and <tE)
    elseif(isequal(opts.pIndex,'wm'))
        %% WMAPE weighted MAPE
        relevance_labels = 1- pdf(opts.pd_model_pmeasure,labels) ./ opts.pd_model_max_pmeasure;
        error =   nansum(relevance_labels .* abs(predictions - labels) ./ (labels) ); %  sum((predictions - labels).^ 2);
        info.relevance(end) = info.relevance(end) + nansum(relevance_labels);
    elseif(isequal(opts.pIndex,'swm'))
        %% sWMAPE s weighted MAPE
        relevance_labels = 1- pdf(opts.pd_model_pmeasure,labels) ./ opts.pd_model_max_pmeasure;
        error =   nansum(relevance_labels .* abs(predictions - labels) ./ (labels + abs(predictions)) ); %  sum((predictions - labels).^ 2);
        info.relevance(end) = info.relevance(end) + nansum(relevance_labels);
    elseif (isequal(opts.pIndex,'mae'))
        %% MAE
        error =   nansum( abs(predictions - labels) ); %  sum((predictions - labels).^ 2);
    elseif (isequal(opts.pIndex,'rmse'))
        %% MSE
        error =   nansum( (predictions - labels).^2 ); %  sum((predictions - labels).^ 2);
    end
    info.error(:,end) = info.error(:,end) + error;
    

end
