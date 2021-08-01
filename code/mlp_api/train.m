function opts = train(X,opts)
%TRAIN Training or Learning for the Perceptron
%           for One epoch

arguments
    X  double {mustBeNumeric}
    opts struct % options structure for training
end

% X is the supervised training data
% and X to be of dimension D x (L+P).
% D is the number of input training patterns or data points
% L is the number of input features including the bias
% P is the number of output features, which is 1 in this case.
%
% opts structure:
% opts.W saved weights
% opts.mode % sgd mode
% opts.N % is the selected number of input training patterns or data points
% opts.L % number of input layer features (nodes)
% opts.P % number of output layer features (nodes)
% opts.D % is the number of input training patterns or data points
% opts.Hwidth % number of hidden layers
% opts.Hnodes % row vector holding number of nodes in each hidden layer
% opts.layer_wdims % weight layer space size
% opts.layers % number of layers in the network (an added + 1 in mlp case).
% opts.epochs % number of epochs to run
% opts.this_epoch % current epoch
% opts.batch_size
% opts.shuffle = logic: 0 or 1
% opts.hessian_search = logic: 0 or 1
% opts.enable_momentum = logic: 0 or 1
% opts.iterations % count iterations, initially 0.


% The number of elements in Hnodes must be equal to Hwidth
if opts.Hwidth~=0
    msg = sprintf("The number of elements in Hnodes must be equal to Hwidth");
    assert(numel(opts.Hnodes) == opts.Hwidth, msg);
end
assert(numel(opts.Hnodes) >= 0, 1);

% network architecture
if opts.Hwidth == 0 % slp
    % weight layer space size
    opts.layer_wdims  = [opts.L, opts.P, opts.P];
    % P X L : initialize weghts -- random
    % opts.W = 2*rand(opts.P, opts.L) - 1;
else % mlp
    % weight layer space size
    % we have added an extraneous unity-buffer layer at the o/p
    % for the sake of uniformity for backprop.
    opts.layer_wdims = [opts.L, opts.Hnodes, opts.P, opts.P];
end
opts.layers = numel(opts.layer_wdims) - 1;
% weights data structure - cell arrays
opts.W = cell(opts.layers,1);

%-------- Weight initialization
s=rng(1); % pseudo-random number generator. reproducible results
for id = 1:opts.layers
    % normal distribution, [0, var]
    c = 2;
    if id ~=opts.layers
        var = sqrt(c/(opts.layer_wdims(id+1)*opts.layer_wdims(id)));
        opts.W{id} = 2*var*rand(opts.layer_wdims(id+1),...
            opts.layer_wdims(id)) - var;
    elseif id == opts.layers - 1
        % uniform distribution, [-var var]
        var = sqrt(c/(opts.layer_wdims(id+1)+opts.layer_wdims(id)));
        opts.W{id} = 2*var*rand(opts.layer_wdims(id+1),...
            opts.layer_wdims(id)) - var;
    elseif id == opts.layers
        % identity matrix (square)
        opts.W{id} = eye(opts.layer_wdims(id+1),...
            opts.layer_wdims(id));
    end
end
rng(s); % pseudo-random number generator

% initialize output gradients
% DY = cell(opts.layers,1);
% output cost
% initialize training error vector
E_tr = zeros(opts.epochs,1);
%-----------

% batch_size setup
if strcmp(opts.mode,'b')
    opts.batch_size = opts.D;
elseif strcmp(opts.mode,'o')
    opts.batch_size = 1;
else
    assert(opts.batch_size >= 1, 'batchsize must be >= 1!');
    assert(opts.batch_size <= opts.D, 'batchsize must be <= opts.D!');
end

opts.this_epoch = 0;
opts.iterations = 0;
for epoch = 1:opts.epochs
    % optional shuffle for each epoch
    if opts.shuffle == 1
        % assumes data array is 2-dimensional
        % with the rows specifying the total number of the data samples
        % and columns representing the features of each of the data sample
        % (row x column)
        % gets the total rows in the data,
        rand_len = randperm(size(X,1));
        % shuffles data by random indexing from the original.
        X = X(rand_len,:);
    end
    minibatches = floor(opts.D/opts.batch_size);
    %
    esum = 0;
    for batch = 1:minibatches
        lowerpt = ((batch-1)*opts.batch_size)+1;
        upperpt = (batch*opts.batch_size);
        if (batch == minibatches) && (mod(opts.D,opts.batch_size)~=0) && ...
                (opts.batch_size~=1) && (opts.batch_size~=opts.D)
            upperpt = upperpt + 1;
        end
        N_batchlen = numel(lowerpt:upperpt); % data length for this batch
        X_tr = X(lowerpt:upperpt,:);
        
        % optional shuffle for each batch
        % if opts.shuffle == 1
        %     % assumes data array is 2-dimensional
        %     % with the rows specifying the total number of the data samples
        %     % and columns representing the features of each of the data sample
        %     % (row x column)
        %     % gets the total rows in the data,
        %     rand_len = randperm(size(X_tr,1));
        %     % shuffles data by random indexing from the original.
        %     X_tr = X_tr(rand_len,:);
        % end
        
        % initialize the weight update sum to zeo
        % sizeW = size(opts.W);
        % dWsum = zeros(sizeW);
        % init. change in weight sum
        sizeW = cell(opts.layers,1);
        dWsum = cell(opts.layers,1);
        for id = 1:opts.layers
            sizeW{id} = size(opts.W{id});
            dWsum{id} = zeros(sizeW{id});
        end
        dWfsum = dWsum;
        
        outlayer = opts.layers - 1;
        xs = cell(outlayer,1);
        y_hats = cell(opts.layers,1);
        dy_hats = cell(opts.layers,1);
        
        % train
        for k = 1:N_batchlen
            
            % extract input and correct output data-points
            xs{1} = X_tr(k,1:opts.L)';
            y = X_tr(k,opts.L+1:opts.L+opts.P)';
            
            % estimated output: y_hat
            % feedforward propagation through the layers
            for j = 1:outlayer
                [y_hats{j},dy_hats{j}] = perceptron(opts.W{j},xs{j});
                if j ~= outlayer
                    xs{j+1} = y_hats{j};
                end
            end
            % compute error
            e = y - y_hats{j};
            dy_hats{j+1} = ones(size(dy_hats{j}));
            y_hats{j+1} = y_hats{j};
            
            % sgd: weight update
            % feedback by backpropagation through the layers
            [opts.W,dWsum,dWfsum] = bpsgd_obm(opts, dWsum, dWfsum,...
                e, y_hats, dy_hats, xs,...
                N_batchlen, k, outlayer);
            
        end % train
        
        opts.iterations = opts.iterations + 1;
        esum = esum + (e.^2)/N_batchlen;
        
    end % batch
    
    % total inference average error for this epoch
    E_tr(epoch,1) = esum;
    % infer training accuracy for this epoch
    Yinf = infer(X,opts);
    Ycorr = X(:,opts.L+1:opts.L+opts.P);
    c_acc = sum(Yinf==Ycorr);
    c_acc_percent = c_acc*100/opts.D;
    % log training progress
    
    if mod(epoch,200)==0 || (epoch==opts.epochs)
        fprintf("Training Progress: %g%% | Accuracy: (%d/%d)=%g%% .\n",...
            100*(epoch/opts.epochs), c_acc, opts.D, c_acc_percent);
    end
    % total iterations = epochs*minibatches;
end

% save trained weights
opts.W{opts.layers} = [];
% save average training error
opts.E_tr = E_tr./opts.epochs;

end
