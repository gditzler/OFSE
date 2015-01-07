clc
clear
close all

alpha = .05;
X = [[5 4 3 1 2];[5 4 1 3 2];[5 1 3 2 4];[5 3 2 4 1];[5 4 2 3 1];[5 3 1 4 2];[5 4 2 3 1];[5 1 3 2 4]];
[N,k] = size(X);
R = mean(X);


chi2 = (12*N)/(k*(k+1))*(sum(R.^2)-k*(k+1)^2/4);
Ff = (N-1)*chi2/(N*(k-1)-chi2);

z = zeros(k,k);
for j = 1:k
  for i = 1:k
    z(j,i) = (R(j)-R(i))/(sqrt(k*(k+1)/(6*N)));
  end
end
pr = normcdf(-z);
pl = normcdf(z);
p2 = 2*normcdf(-abs(z));
pF = 1 - fcdf(Ff,k-1,(k-1)*(N-1)); % pvalue for the f-test
