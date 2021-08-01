function [y,dy] = relu_std(x)
%RELU

arguments
    x double {mustBeNumeric}
end

% output
y = max(0,x);

% first-derivative
dy = double((x > 0));

end