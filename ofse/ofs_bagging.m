function [mistakes, timerz, h_loss] = ofs_bagging(data, labels, opts)
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
%  See also 
%  OFS_BOOSTING, OFS_BAGGING_AVG, OFS_BAGGING_RANDTRUC_AVG

% perform some error checking 
if ~isfield(opts, 'ensemble_size')
  error('opts.ensemble size must be specified');
end
if ~isfield(opts, 'truncate')
  error('opts.truncate must be specified');
end
if ~isfield(opts, 'fixed_partial')
  opts.fixed_partial = false;
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
if ~isfield(opts, 'partial_test')
  opts.partial_test = 0;
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
timerz = 0;

% initialize the OFS models to be sampled from a Gaussian distribution then
% truncate out the vectors. no need to truncate the ensemble model yet 
opts.models = randn(opts.n_features, opts.ensemble_size+1);
mistakes = zeros(length(labels_te), opts.ensemble_size+1);
h_loss = zeros(length(labels_te), opts.ensemble_size+1);

for i = 1:opts.ensemble_size
  opts.models(:, i) = truncate(opts.models(:, i), opts.truncate(i));
end

for t = 1:T-1
  
  if opts.verbose
    if mod(t, 1000) == 0
      disp(['Timestep ', num2str(t), ' of ', num2str(T-1)]);
    end
  end
  
  % if we are testing on partial information then we should create a mask
  % for the features that are available to us
  if opts.partial_test
    mask = zeros(1, opts.n_features);
    q = randperm(opts.n_features);
    mask(q(1:opts.truncate(end))) = 1;
  end
  
  if opts.fixed_partial
    if random('Binomial', 1, opts.epsilon) == 1,
      perm_t = randperm(size(opts.models(:, 1), 1));
      c_t = perm_t(1:opts.truncate(idx)-1);
      v_idx = zeros(size(opts.models(:, 1),1),1);
      v_idx(c_t) = 1;
    else
      v_idx = (opts.models(:, idx)~=0);
    end
  end
  
  for k = 1:opts.ensemble_size
    
    % perform the online bagging update the to `k`th ensmeble member 
    lambda_k = poissrnd(opts.lambda);
    for j = 1:lambda_k
      if opts.fixed_partial
        opts.models(:,k) = update_ofs(data_tr(t, :), labels_tr(t), opts, k, v_idx);
      else
        opts.models(:,k) = update_ofs(data_tr(t, :), labels_tr(t), opts, k);
      end
    end
    
    % predict the output of the `k`th ensmeble member on the testing
    % sequence and update the mistakes if needed. 
    if opts.partial_test
      f_t = opts.models(:,k)'*(data_te(t, :).*mask)';
      if (sign(f_t)*labels_te(t)) < 0
        mistakes(t, k) = 1;
      end
    else
      f_t = opts.models(:,k)'*data_te(t, :)';
      if (sign(f_t)*labels_te(t)) < 0
        mistakes(t, k) = 1;
      end 
    end
    h_loss(t, k) = hinge(f_t, labels_te(t));

  end
  
  % for bagging, average the ensemble models then perform the truncation
  % step, and update the number of mistakes made by the ensemble.
  opts.models(:, end) = truncate(mean(opts.models(:, 1:end-1),2), opts.truncate(end));
  
  tic;
  if opts.partial_test 
    f_t = opts.models(:,end)'*(data_te(t, :).*mask)';
    
    if (sign(f_t)*labels_te(t)) < 0 
      mistakes(t, end) = 1;  
    end
  else
    if (sign(opts.models(:, end)'*data_te(t, :)')*labels_te(t)) < 0 
      mistakes(t, end) = 1;  
    end
  end
  timerz = timerz + toc;
  h_loss(t, end) = hinge(f_t, labels_te(t));
  
  opts.epsilon = opts.epsilon*opts.anneal^t;
end