function [w, b, loss] = weightupdata(lr, w, b, x, y)
logit = sum(sum(w.*x))+b;
loss = sum((logit-y).^2);
delta_b = y-logit;
b = b+lr*delta_b;
delta_w = delta_b*x;
w = w+lr*delta_w;
end