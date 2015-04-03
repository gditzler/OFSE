function weights = update_ofs(x_t, y_t, opts, idx, v_idx)
% UPDATE_OFS Perform a single instance OFS update to parameters
% 
%  weights = UPDATE_OFS(x_t, y_t, opts, idx)
% 
%  @x_t: feature vector
%  @y_t: training label (+/-1)
%  @opts: option structure (see OFS_BAGGING or OFS_BOOSTING)
%  @idx: integer pointing to the ensemble member in opt that needs to be
%  updated with the new training sample
%
% See also OFS_BAGGING, OFS_BOOSTING

if nargin < 5 
  if random('Binomial', 1, opts.epsilon) == 1,
    perm_t = randperm(size(opts.models(:, idx), 1));
    c_t = perm_t(1:opts.truncate(idx)-1);
    v_idx = zeros(size(opts.models(:, idx),1),1);
    v_idx(c_t) = 1;
  else
    v_idx = (opts.models(:, idx)~=0);
  end
end

w_t = opts.models(:, idx);
xt_t = x_t.*v_idx';
f_t = w_t'*xt_t';

% feedback
if y_t*f_t <= 1,
  xh_t = xt_t./((opts.truncate(idx)/numel(x_t))*opts.epsilon + (w_t~=0)'*(1 - opts.epsilon));
  w_temp = w_t + opts.eta*y_t*xh_t';
  w_updates = w_temp*min(1, opts.R/norm(w_temp));
  weights = truncate(w_updates, opts.truncate(idx));
else
  weights = opts.models(:, idx);
end
