function [ u ] = msd_train_error( e )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% e: N-1 or N-nC

if isvector(e) == 0 % one hot --> normal
    e = sum(e, 2);
end
maxe = max(e);
u = 1 - e/(maxe + 1.0e-16);
% u = e/(maxe + 1.0e-16);

% the = mean(e);
% the = 0.9*maxe;
% u = zeros(size(e));
% u(e > the) = 1 - e(e > the)/(maxe + 1.0e-16);
% u(e <= the) = 0.0;
% u(e <= the) = e(e <= the)/(maxe + 1.0e-16);
% u(e <= the) = 1.0;
end

