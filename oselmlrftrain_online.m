function [ net, ers, totaltraining_time ] = oselmlrftrain_online( net, train_X, train_T, test_X, test_T, opts )
%OSELMLRFTRAIN Online Training ELM-LRF
%   
% if N <= K*(d-r+1)^2   beta=H'*pinv(I/C+H*H')*T
% if N > K*(d-r+1)^2    beta=pinv(I/C+H'*H)*H'*T
%
%==========================================================================
% Developed based on "cnn" of "DeepLearnToolbox" of rasmusbergpalm on GitHub
%   https://github.com/rasmusbergpalm/DeepLearnToolbox
%   
%==========================================================================
% ---------<LiuZhi>
% ---------<Xidian University>
% ---------<zhiliu.mind@gmail.com>
% ---------<2015/11/24>
%==========================================================================
%

fprintf('\n-------Online Training-------\n');

if ~isempty(opts.randseed)
    randn('seed', opts.randseed);
end

% timing
totaltraining_time = cputime;

batchSize = opts.batchsize;
epochs = opts.epochs;

N = size(train_X, 3); % since x is H-W-N-C, whatever C is
a = fix(N / batchSize); b = rem(N, batchSize);
if b ~= 0, b = 1; end
numBatches = a + b*1;


% Construct T
train_T = double(train_T); % nSamples-nClasses


% model
elmlrff = str2func(['@oselmlrff_' opts.model]);

ers = zeros(epochs, 1);

for epoch = 1:epochs
    
    ridx = randperm(N, N);
    training_time = cputime;
    
    for l = 1 : numBatches
        idx = (l-1)*batchSize+1 : min(l*batchSize, N);
        bN = length(idx);
        batch_X = train_X( :, :, ridx(idx), : );
        batch_T = train_T(ridx(idx), : );
        % Compute h :batch
        net = elmlrff(net, batch_X, opts);

        Pht = net.P*net.h';
        
        opts.isUseRandErrorFuzzy = 0;
        
        if opts.isUseRandErrorFuzzy > 0
            I = diag(rand(bN,1));
        else
            I = eye(bN, bN);
        end
        
        net.P = net.P - Pht * ((I + net.h*Pht)\(net.h*net.P));
        net.BETA = net.BETA + Pht*(batch_T - net.h*net.BETA);
    end

    [er, ~, ~] = oselmlrftest(net, train_X, train_T, opts);
    ers(epoch) = er;
    training_time = cputime - training_time;

    fprintf('\n~~~~~~~Epoch %d\n', epoch);
    fprintf('\nTraining error: %f\nTraining Time:%fs\n', er, training_time);

    [er, ~, testing_time] = oselmlrftest(net, test_X, test_T, opts);

    fprintf('\nTesting error: %f\nTesting Time:%fs\n', er, testing_time);

end
totaltraining_time = cputime - totaltraining_time;

end

