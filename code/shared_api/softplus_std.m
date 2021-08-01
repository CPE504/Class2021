function [y,dy] = softplus_std(x)
%SOFTPLUS

arguments
    x double {mustBeNumeric}
end

% output
y = log(abs( 1 + (1./exp(-x)) ));

% first-derivative
dy = 1 ./ (1 + exp(-x));

end