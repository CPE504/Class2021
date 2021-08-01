clear *; close all; clc;

%% toy data (linear combinational logic system)
% OR, AND, NOT, NAND, NOR
% nonlinear: XOR/XNOR
X = [ 0 0 1 1;
      0 1 1 0;
      1 0 1 0;
      1 1 1 1; 
    ];
% augmentation 
X = repmat(X,2,1);

%% NN opts data structure:
opts.D = numel(X(:,1)); % is the number of input training patterns or data points
opts.N = 1; % initially selected number of input training patterns or data points

opts.P = 1; % number of output layer features (nodes)
opts.L = numel(X(1,:)) - opts.P; % number of input layer features (nodes)

% number of hidden layers in network
% number of hidden nodes in each hidden layer in network
% -the number of elements in Hnodes must be equal to Hwidth 

% opts.Hwidth = 0; % perceptron
%
opts.Hwidth = 1; % shallow mlp
opts.Hnodes = 4; % shallow
%
% opts.Hwidth = 2; % min is zero, max is a finite number
% opts.Hnodes = [4 2]; % deep

% number of epochs to run
opts.epochs = 10e3; 

% SGD logic: 0 or 1
opts.shuffle = 1; % batch-shuffle (stochastic descent or not) 
opts.hessian_search = 0;
opts.enable_momentum = 0;

%% Training and Inference
%
% disp('Online');
opts.mode = 'o'; % sgd mode
% set batch size
opts.batch_size = 1;
% train
opts = train(X,opts);
Eo = opts.E_tr;
% infer
Yinfo = infer(X,opts);
Ycorr = X(:,4);
display(table(Yinfo,Ycorr));

%
% disp('Batch');
opts.mode = 'b'; % sgd mode
opts.batch_size = opts.D;
% train
opts = train(X,opts);
Eb = opts.E_tr;
% infer
Yinfb = infer(X,opts);
Ycorr = X(:,4);
display(table(Yinfb,Ycorr));

%
disp('Mini Batch');
opts.mode = 'm'; % sgd mode
opts.batch_size = opts.D/2;
% train
opts = train(X,opts);
Em = opts.E_tr;
% infer
Yinfm = infer(X,opts);
Ycorr = X(:,4);
display(table(Yinfm,Ycorr));

%% Visualization
% smooth number of points to display 
smoothid = 100;
% Plot
figure(010);
hf_on = line(1:smoothid:opts.epochs,Eo(1:smoothid:opts.epochs,1),...
'DisplayName', 'online sgd', 'Color', '#f44','Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'r','MarkerIndices',1:opts.epochs:opts.epochs);  %#ok<NASGU>
% hf_on.Color = [0.8 0.8 0.8];
hf_bch = line(1:smoothid:opts.epochs,Eb(1:smoothid:opts.epochs,1),...
    'DisplayName', 'batch sgd', 'Color', '#09f','Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'b','MarkerIndices',1:opts.epochs:opts.epochs);  %#ok<NASGU>
% hf_bch.Color = [0.8 0.8 0.8];
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
%% animate
hf_on = line(1, Eo(1,1), 'Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'r');
hf_bch = line(1, Eb(1,1), 'Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'b');
hf_mb = line(1, Em(1,1), 'Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'g');
% filter legend
lgd.String =  lgd.String(1,1:3); %{'online'  'batch'  'mini-batch'};
%
% animate line of fit: loop through by changing XData and YData
for id = 1:smoothid:opts.epochs
    % data
%     set(hf_on, 'XData', (1:id)', 'YData', Eo(1:id,1),'LineWidth',2); 
    set(hf_on, 'XData', (id)', 'YData', Eo(id,1),'LineWidth',2);
    hold on;
%     set(hf_bch, 'XData', (1:id)', 'YData', Eb(1:id,1),'LineWidth',2); 
    set(hf_bch, 'XData', (id)', 'YData', Eb(id,1),'LineWidth',2);
    
%     set(hf_mb, 'XData', (1:id)', 'YData', Em(1:id,1),'LineWidth',2);  
    set(hf_mb, 'XData', (id)', 'YData', Em(id,1),'LineWidth',2);  
    
    % get frame as an image
    f = getframe(gcf);
    % drawnow limitrate; 
end

