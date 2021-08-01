function [ydrop,dydrop] = dropout(y, dy, r)
% DROPOUT
% r = drop_ratio leaving: 1-ratio

arguments
    y {mustBeNumeric}
    dy {mustBeNumeric}
    r (1,1) {mustBeNumeric}
end

assert(0.0 <= r && r < 1.0,'drop ratio, r must be a fraction less than 1.')
%
n = numel(y);
ydrop = zeros(size(y));
dydrop = zeros(size(dy));

% permutation mask
k = round(n*(1-r));
id_kept = randperm(n, k);

% compensate for drop by 1/(1-r): new y
ydrop(id_kept) = y(id_kept)./(1-r);
dydrop(id_kept) = dy(id_kept)./(1-r);

end