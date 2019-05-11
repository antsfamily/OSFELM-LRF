function [ u, label ] = msd_class_dist( x, y )
%CLASS_CENTER Summary of this function goes here
%   Detailed explanation goes here
% x: H*W*C-N
% y: 1-N or nC-N

if isvector(y) == 0 % one hot --> normal
    [~, y] = max(y);
end

[n, N] = size(x);
label = unique(y);
nC = numel(label);
y = y - min(label) + 1;
newlabel = 1:nC;

s = zeros(n, nC);
num = zeros(1, nC);

for i = 1:N
    k = y(i);
    num(k) = num(k) + 1;
    s(:, k) = s(:, k) + x(:, i);
end

s = bsxfun(@rdivide, s, num);

d = zeros(1, N);
for i = 1:N
    d(i) = norm(x(:, i) - s(:, y(i)));
end

r = zeros(1, nC);
for k = newlabel
    r(k) = max(d(y==k));
end
delta = 1.0e-16;
r = r + delta;

% u = 1 - d/r(y);
thd = mean(d);
idxgt = d > thd;
idxlt = d <= thd;
u = zeros(1, N);
u(idxgt) = 1 - d(idxgt)./r(y(idxgt));
% u(idxlt) = d(idxlt)/r(y(idxlt));
u(idxlt) = 1.0;












