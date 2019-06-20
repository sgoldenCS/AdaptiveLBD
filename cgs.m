function [x,r] = cgs(B,x)
k = size(B,2);
nv = size(x,2);
r = zeros(k+nv,nv);
start = 1;
if isempty(B)
    normx = norm(x(:,1));
    B(:,1) = x(:,1)/normx;
    x(:,1) = B(:,1);
    start = 2;
    r(1,1) = normx;
end

for i = start:nv
    iter = 0;
    normx = 0;
    prev = 1;
    while normx <= 0.7*prev && iter < 4
        iter = iter + 1;
        prev = norm(x(:,i));
        coef = (x(:,i)'*B)';
        x(:,i) = x(:,i) - B*coef;
        r(1:k+i-1,i) = r(1:k+i-1,i) + coef;
        normx = norm(x(:,i));
    end
    if iter > 3
        warning('Vector not orthogonalized!');
    end
    r(k+i,i) = normx;
    x(:,i) = x(:,i)/normx;
    if nv > 1
        B = [B x(:,i)];
    end
end
if iter > 3
    r = [];
end
end
