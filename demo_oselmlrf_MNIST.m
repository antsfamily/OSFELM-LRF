%demo_oselmlrf.m
% A demo of ELM-LRF for MNIST Classiffication
%========================================================================== 
% paper:Huang G, Bai Z, Kasun L, et al. Local Receptive Fields Based 
%   Extreme Learning Machine[J]. Computational Intelligence Magazine IEEE, 
%   2015, 10(2):18 - 29.
%
% myblog:http://blog.csdn.net/enjoyyl/article/details/45724367
%==========================================================================
%
% ---------<Liu Zhi>
% ---------<Xidian University>
% ---------<zhiliu.mind@gmail.com>
% ---------<http://blog.csdn.net/enjoyyl>
% ---------<https://www.linkedin.com/in/%E5%BF%97-%E5%88%98-17b31b91>
% ---------<2015/11/24>
% 

clear all;

%% load MNIST data
data = load('./data/mnist_uint8.mat');

N0 = 20000;
Ns = 60000;
data.train_x = data.train_x(1:Ns, :);
data.train_y = data.train_y(1:Ns, :);
% [ u, label ] = msd_class_dist(double(data.train_x'), data.train_y');

train_x = double(reshape(data.train_x',28,28,Ns))/255;
train_y = data.train_y;

test_x = double(reshape(data.test_x',28,28,10000))/255;
test_y = data.test_y;



%% Setup ELM-LRF
rand('state',0)

oselmlrf.layers = {
	struct('type', 'i') %input layer
	struct('type', 'c', 'outputmaps', 32, 'kernelsize', 5) %convolution layer
	struct('type', 's', 'scale', 3) %sub sampling layer
%     struct('type', 'c', 'outputmaps', 16, 'kernelsize', 4) %convolution layer
% 	struct('type', 's', 'scale', 3) %sub sampling layer
};

opts.isUseClassDistFuzzy = 0;
opts.isUseTrainErrorFuzzy = 0;
opts.isUseRandErrorFuzzy = 0;
opts.batchsize = 10000;
opts.epochs = 30;
opts.model = 'sequential';
% opts.model = 'parallel';
opts.randseed = [];
opts.randseed = 0;
opts.activation = [];
% opts.activation = 'relu';
% opts.activation = 'tanh';

% setup
oselmlrf = oselmlrfsetup(oselmlrf, train_x, opts);

Cs = [ 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
% Cs = [ 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1];
for C = Cs
	opts.C = C;
    fprintf('\n=======With C = %f=======\n', opts.C);
	%% Initial training of ELM-LRF
	[oselmlrf, er, training_time] = oselmlrftrain_Initial(oselmlrf, train_x(:,:,1:N0,:), train_y(1:N0,:), opts);
%     [oselmlrf, er, training_time] = oselmlrftrain_Initial(oselmlrf, train_x, train_y, opts);
	% disp training error
	 fprintf('\nTraining error: %f\nTraining Time:%fs\n', er, training_time);

	%% Test ELM-LRF
	% disp testing error
	[er, bad, testing_time] = oselmlrftest(oselmlrf, test_x, test_y, opts);

	fprintf('\nTesting error: %f\nTesting Time:%fs\n', er, testing_time);
    
    %% Online training of ELM-LRF
    [oselmlrf, ers, training_time] = oselmlrftrain_online(oselmlrf, train_x, train_y, test_x, test_y, opts);
    
end