function [mistakes, timerz] = ofs_bagging_avg(data, labels, opts)
% OFS_BAGGING_AVG Call OFS_BAGGING multiple time using different
% permutations of the sequence in the data. opt.avg field must be set to
% use this function
% 
%  [mistakes, timerz] = OFS_BAGGING_AVG(data, labels, opts)
% 
% See also OFS_BAGGING
if ~isfield(opts, 'avg')
  error('opts.avg must be set');
end
mistakes_c= cell(1, opts.avg);
timerz_c = cell(1, opts.avg);
parfor k = 1:opts.avg 
  ks = randperm(numel(labels));
  data_k = data(ks, :);
  labels_k = labels(ks);
  [mistakes_c{k}, timerz_c{k}] = ofs_bagging(data_k, labels_k, opts);
end


mistakes = zeros(size(mistakes_c{1}));
timerz = zeros(size(timerz_c{1}));

for k = 1:opts.avg 
  mistakes = mistakes + mistakes_c{k};
  timerz = timerz + timerz_c{k};
end
mistakes = mistakes/opts.avg;
timerz = timerz/opts.avg;