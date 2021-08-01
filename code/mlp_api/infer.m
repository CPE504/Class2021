function [Y,E] = infer(X,opts)
%TRAIN Training or Learning for the Perceptron 
%           for One epoch

arguments
    X  double {mustBeNumeric}
    opts struct % options structure for training
end

% see train for elements saved in the opts structure.

outlayer = opts.layers - 1;
xs = cell(outlayer,1);

Y = zeros(opts.D, 1);
E = Y;
% infer
for k = 1:opts.D

    % extract input and correct output data-points
    xs{1} = X(k,1:opts.L)';
    y = X(k,opts.L+1:opts.L+opts.P)';
    
    % estimated output: y_hat
    % feedforward propagation through the layers
    for j = 1:outlayer
        [y_hat] = perceptron(opts.W{j},xs{j});
        if j ~= outlayer
            xs{j+1} = y_hat;
        end
    end
    % decision boundary ~ 0.5
    y_hat(y_hat >= 0.5) = 1;
    y_hat(y_hat < 0.5) = 0;
    e = y - y_hat;
    

    % save instantaneous inference error
    E(k,1) = e;
    % save inference
    Y(k,1) = y_hat;
    
end % infer



end

