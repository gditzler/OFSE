function weights = update_ofs(x_t, y_t, opts, idx)

if random('Binomial', 1, opts.epsilon) == 1,
  perm_t = randperm(size(opts.models(:, idx), 1));
  c_t = perm_t(1:opts.truncate(idx)-1);
  v_idx = zeros(size(opts.models(:, idx),1),1);
  v_idx(c_t) = 1;
  %v_idx = find(v_idx==1);
else
  v_idx = (opts.models(:, idx)~=0);
end

w_t = opts.models(:, idx);
xt_t = x_t.*v_idx';
f_t = w_t'*xt_t';

% feedback
if y_t*f_t<=0,
  xh_t = xt_t/((opts.truncate(idx)/numel(x_t))*opts.epsilon ...
    + sum(opts.models(find(v_idx==1),idx)~=0)*(1-opts.epsilon));
  
  w_temp = w_t + opts.eta*y_t*xh_t';
  w_updates = w_temp*min(1, opts.R/norm(w_temp));
  weights = truncate(w_updates, opts.truncate(idx));
else
  weights = opts.models(:, idx);
end
