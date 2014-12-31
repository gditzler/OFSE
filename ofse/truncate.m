function w_B = truncate(w, B)
% TRUNCATE Truncate the parameter vector 
% 
%  w_B = truncate(w, B) Truncates the l0-norm of w to be a maximum size of B
%
% See also OFS_BAGGING, OFS_BOOSTING, UPDATE_OFS
if sum(w~=0) > B,
    [sw, index] = sort(abs(w),'descend');
    sw(B+1:end) = 0;
    [sw1, index1] = sort(index);
    w_B = sw(index1).*sign(w);
else
    w_B = w;
end
