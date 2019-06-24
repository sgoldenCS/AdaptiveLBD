function [U,S,V,hist] = ABLBD_Main(arg)

A = arg.A; k = arg.k; BlS = arg.BlockSize; Adapt = arg.Adaptive;
Criteria = uint32(arg.Criteria); delta = arg.Delta; alpha = arg.Alpha;
MaxBasis = arg.MaxBasis; MaxMV = arg.MaxMV; MaxTime = arg.MaxTime;
tol = arg.tol;

if ~isempty(arg.Config)
    run(arg.Config);
end

MaxBasis = min([floor((MaxMV+BlS)/2),MaxBasis,min(size(A))]);
assert(BlS < MaxBasis);

if bitand(Criteria,32)
    assert(k < MaxBasis); %Required for finite residual norms
end

af = norm(A,'fro');
mvs = 0;
noexpand = 1;
hist = {};
rnorm = [];
i = 1;
conv = uint32(0);

%colors = lines(iter);

tic

if MaxBasis ~= inf
    V = zeros(size(A,2),MaxBasis);
    U = zeros(size(A,1),MaxBasis);
    B = zeros(MaxBasis,MaxBasis);
end

[V(:,1:BlS),~] = qr(randn(size(A,2),BlS),0);
[U(:,1:BlS),B(1:BlS,1:BlS)] = qr(A*V(:,1:BlS),0); mvs = mvs + BlS;
[ub,sb,vb] = svd(B(1:BlS,1:BlS));
s = diag(sb);
%semilogy(s);

prev = 0; cur = BlS; next = 2*BlS;

fprintf('Time: %f\t Iter: %d\t Matvecs: %d\t k: %d\t BlS: %d\t FroNorm: %d\n',toc,i,mvs,cur,BlS,norm(s)/af);
hist{i} = {toc, i, mvs, cur, BlS, rnorm, norm(s)/af};

while ~bitand(conv,Criteria)
    i = i + 1;
    V(:,cur+1:next) = A'*U(:,prev+1:cur) - V(:,prev+1:cur)*B(prev+1:cur,prev+1:cur);
    mvs = mvs + BlS;
    V(:,cur+1:next) = V(:,cur+1:next) - V(:,1:cur)*(V(:,1:cur)'*V(:,cur+1:next));
    VtV = V(:,cur+1:next)'*V(:,cur+1:next);
    [V(:,cur+1:next),tmp] = qr(V(:,cur+1:next),0);
    B(prev+1:cur,cur+1:next) = tmp';
    
    if Adapt ~= 0
        multiples = max(sum(abs(s - s')./s < Adapt));
        if multiples >= 0.9*BlS && next+BlS < MaxBasis
            V(:,next+1:next+BlS) = cgs(V(:,1:next),randn(size(A,2),BlS));
            next = next + BlS;
            BlS = 2*BlS;
            noexpand = 0;
        end
    end
    
    U(:,cur+1:next) = A*V(:,cur+1:next) - U(:,prev+1:cur)*B(prev+1:cur,cur+1:next);
    mvs = mvs + BlS;
    U(:,cur+1:next) = U(:,cur+1:next) - U(:,1:cur)*(U(:,1:cur)'*U(:,cur+1:next));
    [U(:,cur+1:next),B(cur+1:next,cur+1:next)] = qr(U(:,cur+1:next),0);
    
    [ub,sb,vb] = svd(B(1:next,1:next));
    s = diag(sb);
    
    %hold on; semilogy(s,'Color',colors(floor(next/BlS),:));
    
    if noexpand
        rnorm = sqrt(diag(ub(end-BlS+1:end,:)'*VtV*ub(end-BlS+1:end,:)));
        %    semilogy(s+run,'Color',colors(floor(next/BlS),:));
    end
    
    fprintf('Time: %f\t Iter: %d\t Matvecs: %d\t k: %d\t BlS: %d\t FroNorm: %d\n',toc,i,mvs,next,BlS,norm(s)/af);
    hist{i} = {toc, i, mvs, next, BlS, rnorm, norm(s)/af};
    
    if bitand(Criteria,1) && norm(s)/af > delta
        conv = bitor(conv, 1);
    end
    if bitand(Criteria,2)
        ind = find(rnorm+s(1:size(rnorm)) < alpha*s(1),1);
        if rnorm(ind) < delta
            conv = bitor(conv, 2);
        end
    end
    if bitand(Criteria,4) && mvs > MaxMV
        conv = bitor(conv, 4);
    end
    if bitand(Criteria,8) && toc > MaxTime
        conv = bitor(conv, 8);
    end
    if bitand(Criteria,16) && next >= MaxBasis
        conv = bitor(conv, 16);
    end
    if bitand(Criteria,32) && all(rnorm(1:k) < tol)
        conv = bitor(conv, 32);
    end
    %Add local gap criteria
    
    prev = cur;
    cur = next;
    next = next + BlS;
end

U = U(:,1:cur)*ub(:,1:min(k,cur));
S = sb(1:min(k,cur),1:min(k,cur));
V = V(:,1:cur)*vb(:,1:min(k,cur));

end