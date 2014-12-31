function [mistakes, timerz] = ofs_bagging(data, labels, opts)
% OFS_BAGGING Online Bagging using Online Feature Selection 
%
%  [mistakes, timerz] = OFS_BAGGING(data, labels, opts)
%
%  @data: feature vectors (observations by features)
%  @labels: class labels (+/-1)
%  @opt: options structure
%     @opts.lambda: Poisson parameter for bagging
%     @opts.ensemble_size: ensemble size (required)
%     @opts.epsilon: search term
%     @opts.eta: learning rate
%     @opts.R: max l2-norm
%     @opts.truncate: l0-norm (required)
%     @opts.verbose: print evaulation round (mod(t,1000)==0)
%
%  See also OFS_BOOSTING, OFS_BAGGING_AVG, OFS_BAGGING_RANDTRUC_AVG

% perform some error checking 
if ~isfield(opts, 'ensemble_size')
  error('opts.ensemble size must be specified');
end
if ~isfield(opts, 'truncate')
  error('opts.truncate must be specified');
end
if ~isfield(opts, 'epsilon')
  opts.epsilon = .1;
end
if ~isfield(opts, 'eta')
  opts.eta = .1;
end
if ~isfield(opts, 'lambda')
  opts.lamba = 1;
end
if ~isfield(opts, 'verbose')
  opts.verbose = 0;
end
if length(opts.truncate) == 1
  opts.truncate = opts.truncate*ones(1, opts.ensemble_size+1);
end
if length(opts.truncate) ~= (opts.ensemble_size+1)
  error('opts.truncate must be of length opts.ensemble_size+1.')
end
if ~isfield(opts, 'anneal')
  opts.anneal = 1;
end
[T, opts.n_features] = size(data);

% split the data into train/testing sequences assuing 
data_tr = data(1:T-1, :);
data_te = data(2:T, :);
labels_tr = labels(1:T-1);
labels_te = labels(2:T);

% initialize the OFS models to be sampled from a Gaussian distribution then
% truncate out the vectors. no need to truncate the ensemble model yet 
opts.models = randn(opts.n_features, opts.ensemble_size+1);
mistakes = zeros(length(labels_te), opts.ensemble_size+1);
for i = 1:opts.ensemble_size
  opts.models(:, i) = truncate(opts.models(:, i), opts.truncate(i));
end

tic;
for t = 1:T-1
  
  if opts.verbose
    if mod(t, 1000) == 0
      disp(['Timestep ', num2str(t), ' of ', num2str(T-1)]);
    end
  end
  
  for k = 1:opts.ensemble_size
    
    % perform the online bagging update the to `k`th ensmeble member 
    lambda_k = poissrnd(opts.lambda);
    for j = 1:lambda_k
      opts.models(:,k) = update_ofs(data_tr(t, :), labels_tr(t), opts, k);
    end
    
    % predict the output of the `k`th ensmeble member on the testing
    % sequence and update the mistakes if needed. 
    if (sign(opts.models(:,k)'*data_te(t, :)')*labels_te(t)) < 0
      mistakes(t, k) = 1;
    end  

  end
  
  % for bagging, average the ensemble models then perform the truncation
  % step, and update the number of mistakes made by the ensemble.
  opts.models(:, end) = truncate(mean(opts.models(:, 1:end-1),2), opts.truncate(end));
  if (sign(opts.models(:, end)'*data_te(t, :)')*labels_te(t)) < 0 
    mistakes(t, end) = 1;  
  end
  opts.epsilon = opts.epsilon*opts.anneal^t;
end
timerz = toc;