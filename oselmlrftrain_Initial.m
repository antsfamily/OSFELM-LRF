function [ net, er, training_time ] = oselmlrftrain_Initial( net, x, y, opts )
%ELMLRFTRAIN Train ELM-LRF
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

fprintf('\n-------Initial Training-------\n');

if ~isempty(opts.randseed)
    randn('seed', opts.randseed);
end

% timing
training_time = cputime;

batchSize = opts.batchsize;

N = size(x, 3); % since x is H-W-N-C, whatever C is
a = fix(N / batchSize); b = rem(N, batchSize);
if b ~= 0, b = 1; end
numBatches = a + b*1;

% K = net.layers{end}
% H = zeros(N, K*(d-r+1)^2);
H = [];

% model
elmlrff = str2func(['@oselmlrff_' opts.model]);

for l = 1 : numBatches
    idx = (l-1)*batchSize+1 : min(l*batchSize, N);
    batch_x = x( :, :, idx, : );
    % Compute h :batch
    net = elmlrff(net, batch_x, opts);
    % Combine H
    H = cat(1, H, net.h);

end

clear x batch_x kk idx idxkk;



% Construct T
T = double(y); % nSamples-nClasses
clear y;

[N, L] = size(H);

if opts.isUseRandErrorFuzzy > 0
    if N <= L
        I = diag(1./rand(N,1));
    else
        I = diag(1./rand(L,1));
    end
else
        if N <= L
            I = eye(N, N);
        else
            I = eye(L, L);
        end
end

if N <= L  % H is [N, K*(d-r+1)^2]
    net.BETA = H' * ((I/opts.C + H*H') \ T); % A*inv(B)*C  --> A*(B\C)
else
    net.P = inv(I/opts.C + H'*H);
    net.BETA = net.P * H' * T; 
end


[~, label0] = max(T, [], 2);
[~, label] = max(H * net.BETA, [], 2);

bad = find(label0 ~= label);
er = numel(bad) / N;

training_time = cputime - training_time;
end

