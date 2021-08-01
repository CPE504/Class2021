function [y,dy] = lsig_std(x)
%LSIG_STD Simple Logistic-Sigmoid

arguments
    x double {mustBeNumeric}
end

% output
y = 1 ./ (1 + exp(-x));

% first-derivative
dy = y.*(1-y);

end


% y = (2./(1 + exp(-2.*x))) - 1;
% dy = y - y.^2;