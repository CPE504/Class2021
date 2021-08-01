function opts = train(X,opts)
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
% opts.W saved weights
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

% initialize training error vector
E_tr = zeros(opts.epochs,1);


% batch_size setup
if strcmp(opts.mode,'b')
    opts.batch_size = opts.D;
elseif strcmp(opts.mode,'o')
    opts.batch_size = 1;
else
    assert(opts.batch_size > 1, 'batchsize must be > 1!');
    assert(opts.batch_size < opts.D, 'batchsize must be < opts.D!');
end

for epoch = 1:opts.epochs
%     rand_len = randperm(size(X,1));
%     % shuffles data by random indexing from the original.
%     X = X(rand_len,:);
    
    minibatches = floor(opts.D/opts.batch_size);
    %
    for batch = 1:minibatches
        lowerpt = ((batch-1)*opts.batch_size)+1;
        upperpt = (batch*opts.batch_size);
        if (batch == minibatches) && (mod(opts.D,opts.batch_size)~=0) && ...
                (opts.batch_size~=1) && (opts.batch_size~=opts.D)
            upperpt = upperpt + 1;
        end
        batchN = numel(lowerpt:upperpt); % data length for this batch
        X_tr = X(lowerpt:upperpt,:);
        
        % optional shuffle for each batch
        if opts.shuffle == 1
            % assumes data array is 2-dimensional
            % with the rows specifying the total number of the data samples
            % and columns representing the features of each of the data sample
            % (row x column)
            % gets the total rows in the data,
            rand_len = randperm(size(X_tr,1));
            % shuffles data by random indexing from the original.
            X_tr = X_tr(rand_len,:);
        end
        % initialize the weight update sum to zeo
        sizeW = size(opts.W);
        dWsum = zeros(sizeW);
        
        % train
        for k = 1:batchN
            
            % extract input and correct output data-points
            x = X_tr(k,1:opts.L)';
            y = X_tr(k,opts.L+1:opts.L+opts.P)';
            
            % estimated output
            [y_hat,dy_hat] = perceptron(opts.W,x);
            e = y - y_hat;
            [opts.W,dWsum] = sgd_obm(opts.W, dWsum,...
                opts.alpha, e, dy_hat, x,...
                batchN, k, opts.mode);
            
        end % train
        
        opts.iterations = opts.iterations + 1;
        
    end % batch
        
    % total inference error for this epoch
    E_tr(epoch,1) = e;
    % check inference training accuracy for this epoch 
    % infer
    Yinf = infer(X,opts);
    Ycorr = X(:,opts.L+1:opts.L+opts.P);
    c_acc = sum(Yinf==Ycorr);
    c_acc_percent = c_acc*100/opts.D;
    % log training progress
    if mod(epoch,100)==0 || (epoch==opts.epochs)
        fprintf("Training Progress: %g%% | Accuracy: (%d/%d)=%g%% .\n",...
            100*(epoch/opts.epochs), c_acc, opts.D, c_acc_percent);
    end
    
    % total iterations = epochs*minibatches;
end

% save trained weights
% opts.W;
% save average training error
opts.E_tr = E_tr./opts.epochs;

end
