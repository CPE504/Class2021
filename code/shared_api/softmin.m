function [y,dy] = softmin(x)
%SOFTMAX

arguments
    x double {mustBeNumeric}
end

% output
% softmin
% y = (1-0).*exp(-x)./sum(exp(-x));
Q = sum(exp(-x)) - exp(-x);
Qe = Q.*exp(x);
y = 1./(1+Qe);

% first-derivative
dy = y.*(1-y);

end