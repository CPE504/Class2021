function [W,dWsum] = sgd_obm(W,dWsum,alpha,e,dys,xs,N,k,mode)
%SGD_OBM Stochastic Gradient Descent with 3 modes
%   O-Online, B-Batch, M-MiniBatch

arguments
    W double
    dWsum double
    alpha double {mustBeNumeric}
    e double {mustBeNumeric}
    dys double
    xs double
    N (1,1) double {mustBeNumeric}
    k (1,1) double {mustBeNumeric}    
    mode string {mustBeNonzeroLengthText} = 'o'
end


delta = e.*dys;
dW = alpha.*delta.*xs;
dW = dW';

if strcmp(mode,'o')
    dWsum = dW;
    W = W + dWsum;
elseif strcmp(mode,'b') || strcmp(mode,'m')
    dWsum = dWsum + dW;
end




% N == 1 for mode 'o'
% N > 1 for mode 'b' 
% N >= 1 for 'm'

% update: done after all partial dervatives: dE/dW are computed
if k == N
    W = W + (dWsum./N);
end

end

