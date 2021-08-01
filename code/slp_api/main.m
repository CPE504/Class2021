clear *; close all; clc;

% data (linear combinational logic system)
% OR, AND, NOT, NAND, NOR
% nonlinear: XOR
X = [ 0 0 1 0;
      0 1 1 1;
      1 0 1 1;
      1 1 1 1;
    ];

s = rng(5); % pseudo-random number generator. reproducible results

% NN opts data structure:
opts.alpha = 0.9; % sgd learning rate or step-size
opts.D = numel(X(:,1)); % is the number of input training patterns or data points
opts.N = 1; % initially selected number of input training patterns or data points
opts.P = 1; % number of output layer features (nodes)
opts.L = numel(X(1,:)) - opts.P; % number of input layer features (nodes)
opts.shuffle = 0; % shuffle logic: 0 or 1
opts.this_epoch = 0; % current epoch
opts.iterations = 0; % count iterations, initially 0. 
% 1 x 3 here
opts.Wls = [opts.L, opts.P]; % weight layer space size
opts.epochs = 1e3; % number of epochs to run

%%
%
disp('Online');
opts.mode = 'o'; % sgd mode
% initialize weghts -- random
opts.W = 2*rand(opts.P, opts.L) - 1; 
% set batch size
opts.batch_size = 1;

% train
opts = train(X,opts);
Eo = opts.E_tr;
% infer
Yinf = infer(X,opts);
Ycorr = X(:,4);
display(table(int32(Yinf),Ycorr));

%
disp('Batch');
opts.mode = 'b'; % sgd mode
opts.W = 2*rand(opts.P, opts.L) - 1;
opts.batch_size = opts.D;

% train
opts = train(X,opts);
Eb = opts.E_tr;
% infer
Yinf = infer(X,opts);
Ycorr = X(:,4);
display(table(int32(Yinf),Ycorr));

%
disp('Mini Batch');
opts.mode = 'm'; % sgd mode
opts.W = 2*rand(opts.P, opts.L) - 1;
opts.batch_size = opts.D/2;

% train
opts = train(X,opts);
Em = opts.E_tr;
% infer
Yinf = infer(X,opts);
Ycorr = X(:,4);
display(table(int32(Yinf),Ycorr));

%% Visualization

figure(010);
hf_on = line(1:opts.epochs,Eo(1:opts.epochs,1),...
'DisplayName', 'online sgd', 'Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'r','MarkerIndices',1:opts.epochs:opts.epochs); 
hf_on.Color = [0.8 0.8 0.8];
hf_bch = line(1:opts.epochs,Eb(1:opts.epochs,1),...
    'DisplayName', 'batch sgd', 'Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'b','MarkerIndices',1:opts.epochs:opts.epochs); 
hf_bch.Color = [0.8 0.8 0.8];
hf_mb = line(1:opts.epochs,Em(1:opts.epochs,1),...
    'DisplayName', 'mini-batch sgd', 'Marker', '.', 'MarkerSize', 20, ...
    'MarkerEdgeColor', 'g','MarkerIndices',1:opts.epochs:opts.epochs);
hf_mb.Color = [0.8 0.8 0.8];
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
for id = 1:opts.epochs
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


