clear *; close all; clc;

%% toy data (linear combinational logic system)
% OR, AND, NOT, NAND, NOR
% nonlinear: XOR/XNOR
X = [ 0 0 1 0 0 0 1; % 0 class 1, 0 0 0 1
      0 1 1 0 0 1 0; % 1 class 2, 0 0 1 0
      1 0 1 0 0 1 0; % 1 class 2, 0 0 1 0
      1 1 1 0 0 0 1; % 0 class 1, 0 0 0 1
    ];
% augmentation 
X = repmat(X,3,1);

%% NN opts data structure:
opts.D = numel(X(:,1)); % is the number of input training patterns or data points
opts.N = 1; % initially selected number of input training patterns or data points

opts.P = 4; % number of output layer features (nodes)
opts.L = numel(X(1,:)) - opts.P; % number of input layer features (nodes)

% number of hidden layers in network
% number of hidden nodes in each hidden layer in network
% -the number of elements in Hnodes must be equal to Hwidth 

% opts.Hwidth = 0; % perceptron
%
% opts.Hwidth = 1; % shallow mlp
% opts.Hnodes = 4; % shallow

opts.Hwidth = 2; % min is zero, max is a finite number
opts.Hnodes = [50 25]; % deep
opts.hidactv = "relu";
opts.outactv = "softmax";
opts.dropratio = 0.2;

% number of epochs to run
opts.epochs = 5e3; 

% SGD logic: 0 or 1
opts.shuffle = 1; % batch-shuffle (stochastic descent or not) 
opts.hessian_search = 1;
opts.enable_momentum = 1;

%% Training and Inference
%
disp('Mini Batch');
opts.mode = 'm'; % sgd mode
opts.batch_size = opts.D/2;
% train
opts = train(X,opts);
Em = opts.E_tr;
% infer
Yinfm = infer(X,opts);
Ycorr = X(:,opts.L+1:opts.L+opts.P);
display(table(Yinfm,Ycorr));

%% Visualization
% smooth number of points to display 
smoothid = 100;
% Plot
figure(010);
hf_mb = line(1:smoothid:opts.epochs,Em(1:smoothid:opts.epochs,1),...
    'DisplayName', 'mini-batch sgd', 'Color', '#0f9','Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'g','MarkerIndices',1:opts.epochs:opts.epochs); %#ok<NASGU>
% hf_mb.Color = [0.8 0.8 0.8];
xlabel('epoch',...
    'Interpreter','tex','FontName','Consolas','FontSize',10)
ylabel('average training error',...
    'Interpreter','tex','FontName','Consolas','FontSize',10);
lgd = legend('location','best',...
    'Interpreter','tex','FontName','Consolas','FontSize',9);
axis padded;
hold on;

%% Animate
hf_mb = line(1, Em(1,1), 'Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'c');
% filter legend
lgd.String =  lgd.String(1,1); %{'mini-batch'};
%
% animate line of fit: loop through by changing XData and YData
for id = 1:smoothid:opts.epochs
    % data
%     set(hf_mb, 'XData', (1:id)', 'YData', Em(1:id,1),'LineWidth',2);  
    set(hf_mb, 'XData', (id)', 'YData', Em(id,1),'LineWidth',2);  
    
    % get frame as an image
    f = getframe(gcf);
    % drawnow limitrate; 
end

