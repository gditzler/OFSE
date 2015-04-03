function c = csum(x)

c = zeros(size(x, 1), size(x, 2));
for i = 1:size(x, 1)
  if i == 1
    c(i, :) = x(i, :);
  else
    c(i, :) = x(i, :)+c(i-1, :);
  end
end
