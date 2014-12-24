function [ params] = update_ofs(x_t, y_t, params, NumFeature)
%UPDATEOFS Updates the current selector with new features
%   Detailed explanation goes here
%err_count = 0;
epsilon = 0.2;
eta = 0.2;
R = 10;
z_t=random('Binomial',1,epsilon);

if z_t==1,
  perm_t=randperm(size(params,1));
  c_t=perm_t(1:NumFeature-1);
  v_idx=zeros(size(params,1),1);
  v_idx(c_t)=1;
else
  v_idx=(params~=0);
end

xt_t=x_t.*v_idx;
% prediction
f_t=params'*xt_t;
% feedback
if y_t*f_t<=0,
  % err_count = err_count + 1;
  xh_t=xt_t./(NumFeature/size(params,1)*epsilon+(params~=0)*(1-epsilon));
  w_temp=params+eta*y_t*xh_t;
  % projection
  params = w_temp*min(1,R/norm(w_temp));
  params  = truncate(params,NumFeature);
end

