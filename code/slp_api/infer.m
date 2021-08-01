function [Y,E] = infer(X,opts)
%TRAIN Training or Learning for the Perceptron 
%           for One epoch

arguments
    X  double {mustBeNumeric}
    opts % options structure for training
end

% For this case: P = 1
% expects opts.W  to be of dimension L x P
% and X to be of dimension D x (L+P). 
% D is the number of input training patterns or data points
% L is the number of input features including the bias
% P is the number of output features, which is 1 in this case.
%
% opts structure:
% opts.alpha % sgd learning rate or step-size
% opts.mode % sgd mode
% opts.N % is the selected number of input training patterns or data points
% opts.L % number of input layer features (nodes)
% opts.P % number of output layer features (nodes)
% opts.D % is the number of input training patterns or data points
% opts.Wls % weight layer space size
% opts.epochs % number of epochs to run
% opts.this_epoch % current epoch
% opts.batch_size
% opts.shuffle % shuffle logic: 0 or 1
% opts.iterations % count iterations, initially 0. 



Y = zeros(opts.D, 1);
E = Y;
% infer
for k = 1:opts.D
    
    % extract input and correct output data-points
    x = X(k,1:opts.L)';
    y = X(k,opts.L+1:opts.L+opts.P)';
    
    % estimated output
    y_hat = perceptron(opts.W,x);
    e = y - y_hat;

    % decision boundary ~ 0.5 for binary classification
    y_hat(y_hat >= 0.5) = 1;
    y_hat(y_hat < 0.5) = 0;
    e = y - y_hat;
    
    
    % save inference error
    E(k,1) = e;
    % save inference
    Y(k,1) = y_hat;
    
end % infer



end

