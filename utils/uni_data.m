function uni_data(n_samples, n_features, n_relevent, fout)
% [data,labels] = uni_data(n_samples, n_features, n_relevent)
% 
%   @n_samples - number on data points
%   @n_features - total number of features
%   @n_relevent - number of relevant features (< @n_features)
%   @fout - file output
% 
%   Generate a data set which has feature uniformly distributed. There 
%   are only @n_relevent features in the data set that carry information
%   about the class label. 
%   
%  Written by: Gregory Ditzler (gregory.ditzler@gmail.com)  
data = round(10 * rand(n_samples, n_features));
T = 5 * n_relevent;
labels = zeros(n_samples, 1);

labels(sum(data(:,1:n_relevent),2) > T) = -1;
labels(labels == 0) = 1;

i = randperm(n_samples);
i = i(1:floor(.05*n_samples));
labels(i) = -1*labels(i);  % add noise

data = [labels,data];
save(fout, 'data');
