function [y,dy] = perceptron(W,x,type)
%PERCEPTRON Single-Layer NN

arguments
    W  double {mustBeNumeric}
    x  double {mustBeNumeric}
    type = "lsig"
end
% expects W and x to be of the same dimensions L x 1. 


% weighted sum: matrix multiplication
v = W*x;
% activation

if strcmp(type,"lsig")
[y,dy]= lsig_std(v);
elseif strcmp(type,"softmax")
[y,dy]= softmax(v);
elseif strcmp(type,"relu")
[y,dy]= relu_std(v);
elseif strcmp(type,"softmin")
[y,dy]= softmin(v);
elseif strcmp(type,"softplus")
[y,dy]= softplus_std(v);
%else
end


end

