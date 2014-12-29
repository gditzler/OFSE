function [Y,X] = standardize_data(data)
%standardizeData Standardizes data for pre-processing

[n,d] = size(data); % n=rows, d=columns
Y = data(1:n,1);    % Y = all data in first col
X = data(1:n,2:d);  % X = all data except for the first col

stdX = std(X);                            %stdX = standard deviation of X
idx1 = stdX~=0;                           %idx1 persists where the std X is not zero
centrX = X-repmat(mean(X),size(X,1),1);   %Subtracts a tiling from X
X(:,idx1) = centrX(:,idx1)./repmat(stdX(:,idx1),size(X,1),1); %$X at the specified indices = centrX at the same indices, divided by a tiled matrix of the standard deviation at the specified indices

X = (X-repmat(mean(X),size(X,1),1))./repmat(std(X),size(X,1),1); %Overwrite X as current X-tiled mean
X = X./repmat(sqrt(sum(X.*X,2)),1, size(X,2));    %Overwrite X as x X divided by tiled sqrt(sum of X.*X,2)

i = randperm(numel(Y));
Y = Y(i);
X = X(i, :);
