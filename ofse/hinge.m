function loss = hinge(f, y)
loss = max([0, 1 - f*y]);