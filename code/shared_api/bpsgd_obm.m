function [W,dWsum,dWfsum] = bpsgd_obm(opts,dWsum,dWfsum,e,ys,dys,xs,N,k,outlayer)
%SGD_OBM Stochastic Gradient Descent with 3 modes
%   O-Online, B-Batch, M-MiniBatch
% (c.) 2021. oasomefun@futa.edu.ng

arguments
    opts
    dWsum cell
    dWfsum cell
    e double {mustBeNumeric}
    ys cell
    dys cell
    xs cell
    N (1,1) double {mustBeNumeric}
    k (1,1) double {mustBeNumeric}    
    outlayer (1,1) double {mustBeNumeric}
end

mustBeNonzeroLengthText(opts.mode);
W = opts.W;
mode = opts.mode;

% w-lambda, regularizing weight/penalty is typically a small value 
% tending to zero to prevent both underfitting and overfitting
w_lambda = 1e-4; % 1e-2, 1e-3, 1e-4, 1e-6, 1e-8
% epsval: small constant to prevent divison by zero
epsval = 1e-8;

% alpha % sgd learning rate or step-size
alpha = 1;
% filtering/momentum constant
beta = 0.9;

j = outlayer;

% delta from the invisible buffer/unity layer.
% dys{j+1} and W{j+1} are both equal to ones
% e_out = e_lambda*e;
% deltas{j+1} = dys{j+1}.*e_out;

% loglsq: generalization of cross-entropy 
% lsq;
% e_lambda = 1;
%
% crossent or loglsq
% e_lambda = 1./dys{j}; 
% - the above e_lambda is equivalent to dividing out 
% the dys{j=outlayer} in the delta formula at the j=outlayer
% - to avoid division by 0, we can reparameterize as
% e_lambda = 1; and then set dys{j=outlayer} to a vector of 1;
% dys{j} = dys{j+1};

loss = "loglsq"; % recommended for stochastic update
% loss = "lsq";
if strcmp(loss,"loglsq")
    e_lambda = 1;
    dys{j} = dys{j+1};
else
    e_lambda = 1;
end

deltas = dys{j+1}.*(e_lambda.*e); 
%% Backward feedback propagation:
for j = outlayer:-1:1
    
    %
    % errs{j} = (W{j+1})'*deltas{j+1};
    % deltas{j} = dys{j}.*errs{j};
    % dW{j} = alpha.*deltas{j}.*xs{j}';
    % dW{j} = dW{j}';
    
    % errors and deltas
    errs = (W{j+1})'*deltas;
    deltas = dys{j}.*(errs);
    
    % second-order preconditioner (hessian)
    if opts.hessian_search  == 1
        deltak = ((errs.*(1 - (2.*ys{j}))) - (dys{j}));
        ddeltasw = deltak.*(-dys{j}*xs{j}');
        h = (ddeltasw).*xs{j}';
        h = h + norm(h,'fro');
        hinv = norm(h,'fro')./(h);
    else
        % std. gradient descent choice
        hinv = 1.0; % identity = ones(size(deltas)); 
    end
    % line search step: gradient descent or integral-control
	dW = alpha.*(hinv).*(deltas*xs{j}');
    dW = dW./(norm(dW) + epsval ); % L2-norm regularization for dW 
    % isnan(dW)
  
    % undecoupled L2-norm regularization for W 
    % dW = dW + ((w_lambda.*rand(size(W{j}))).*W{j}./(norm(W{j})));
     
    if strcmp(mode,'o')
        dWsum{j} = dW;
    elseif strcmp(mode,'b') || strcmp(mode,'m')
        dWsum{j} = dWsum{j} + dW;
    end
end

% delta = e.*dy; 
% dW = alpha.*delta.*xs;
% dW = dW'; 

% N == 1 for mode 'o'
% N > 1 for mode 'b' 
% N >= 1 for 'm'

%% Gradient descent update: 
% done after all partial dervatives: dE/dW are computed
% for all layers.
if k == N
    for j = 1:outlayer
        
        % filtering/momentum
        if opts.enable_momentum == 1
            %dWfsum{j} = dWfsum{j} + beta.*(dWsum{j}-dWfsum{j});
            dWfsum{j} = beta.*dWfsum{j} + dWsum{j};
        else
            dWfsum{j} = dWsum{j};
        end
        
        % W{j} = W{j} + dWfsum{j};
        % decoupled L2-norm regularization
        W{j} = ( ones(size(W{j})) + ...
            ((w_lambda.*rand(size(W{j})))./(norm(W{j})))).*W{j} ...
            + (dWfsum{j}./N);
    end
end

end

