function mat2arff(X, y, output_fp, relation)
% 
% @param X data (n x F)
% @param y labels (n x 1)
% @param output_fp string pointing to the output file
% @param relation relation for the arff file 
%
% >> load splice
% >> mat2arff(data(:, 2:end), data(:, 1), 'test.arff', 'splice')
%
addpath('../ofse/');
fid = fopen(output_fp, 'w');

data = [y, X];
[y, X] = standardize_data(data);
[n_obs, n_feat] = size(X);

fprintf(fid, ['@relation ', relation, '\n\n']);

for i = 1:n_feat
  fprintf(fid, ['@attribute attribute', num2str(i), ' numeric\n']);
end
fprintf(fid, ['@attribute class {1,-1}\n\n\n@data\n']);

for n = 1:n_obs
  for i = 1:n_feat
    fprintf(fid, [num2str(X(n, i)), ',']);
  end
  fprintf(fid, [num2str(y(n)), '\n']);
end


fclose(fid);