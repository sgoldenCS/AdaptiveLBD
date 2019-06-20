function [U,S,V,hist] = ABLBD_Main(arg)

A = arg.A; k = arg.k; BlS = arg.BlockSize; Adapt = arg.Adaptive;
MaxMV = arg.MaxMV; MaxBasis = arg.MaxBasis; MaxTime = arg.MaxTime;
Criteria = arg.Criteria; Args = arg.CriteriaArgs;

MaxBasis = min([floor((MaxMV+BlS)/2),MaxBasis,min(size(A))]);
assert(BlS < MaxBasis);
if k < 0
    k = MaxBasis;
end

af = norm(A,'fro');
mvs = 0;
noexpand = 1;
hist = {};
run = [];
i = 1;

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

fprintf('Time: %f\t Iter: %d\t Matvecs: %d\t k: %d\t BlS: %d\t FroNorm: %d\n',toc,i,mvs,next,BlS,norm(s)/af);
hist{i} = {toc, i, mvs, next, BlS, run, norm(s)/af};

while mvs < MaxMV && toc < MaxTime && next <= MaxBasis
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
        run = sqrt(diag(ub(end-BlS+1:end,:)'*VtV*ub(end-BlS+1:end,:)));
        %    semilogy(s+run,'Color',colors(floor(next/BlS),:));
    end
    
    fprintf('Time: %f\t Iter: %d\t Matvecs: %d\t k: %d\t BlS: %d\t FroNorm: %d\n',toc,i,mvs,next,BlS,norm(s)/af);
    hist{i} = {toc, i, mvs, next, BlS, run, norm(s)/af};
    
    switch Criteria
        case 1
            if norm(s)/af > Args.delta
                break;
            end
    end
    
    prev = cur;
    cur = next;
    next = next + BlS;
end
U = U(:,1:cur)*ub(:,1:min(k,cur));
S = sb(1:min(k,cur),1:min(k,cur));
V = V(:,1:cur)*vb(:,1:min(k,cur));
end