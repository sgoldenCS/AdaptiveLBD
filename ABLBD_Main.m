function [U,S,V,hist] = ABLBD_Main(arg)

A = arg.A; k = arg.k; BlS = arg.BlockSize; Adapt = arg.Adaptive;
delta = arg.Delta; alpha = arg.Alpha;
MaxTime = arg.MaxTime;
tol = arg.tol; mode = arg.Mode;

if ~isempty(arg.Config)
    run(arg.Config);
end

[MaxMV,MaxBasis,iters] = ...
    setComputationLength(arg.MaxMV,arg.MaxBasis,arg.MaxIter,BlS);

af = norm(A,'fro');
mvs = 0;
noexpand = 1;
hist = {};
rnorm = [];
i = 1;

tic

if MaxBasis ~= inf
    V = zeros(size(A,2),MaxBasis);
    U = zeros(size(A,1),MaxBasis);
    B = zeros(MaxBasis,MaxBasis);
end

[V(:,1:BlS),~] = qr(randn(size(A,2),BlS),0);
[U(:,1:BlS),B(1:BlS,1:BlS)] = qr(A*V(:,1:BlS),0); mvs = mvs + BlS;
R(1:BlS,1:BlS) = B(1:BlS,1:BlS);
[ub,sb,vb] = svd(B(1:BlS,1:BlS));
s = diag(sb);

prev = 0; cur = BlS; next = 2*BlS;

fprintf('Time: %f\t Iter: %d\t Matvecs: %d\t k: %d\t BlS: %d\t FroNorm: %d\n',toc,i,mvs,cur,BlS,norm(s)/af);
hist{i} = {toc, i, mvs, cur, BlS, rnorm, norm(s)/af};

while 1
    noexpand = 1;
    i = i + 1;
    
    V(:,cur+1:next) = A'*U(:,prev+1:cur); mvs = mvs + BlS;
    [V(:,cur+1:next),tmp] = cgs(V(:,1:cur),V(:,cur+1:next));
    vr = tmp(cur+1:next,:);
    B(prev+1:cur,cur+1:next) = vr';
    RtR = vr'*vr;
    
    %multiples = max(sum(abs(s - s')./s < Adapt)); %For newer versions
    multiples = max(sum(bsxfun(@rdivide,abs(bsxfun(@minus,s,s')),s) < Adapt));
    if Adapt ~= 0
        if multiples >= 0.9*BlS
            V(:,next+1:next+BlS) = cgs(V(:,1:next),randn(size(A,2),BlS));
            next = next + BlS;
            BlS = 2*BlS;
            noexpand = 0;
        end
    end
    
    U(:,cur+1:next) = A*V(:,cur+1:next); mvs = mvs + BlS;
    [U(:,cur+1:next),tmp] = cgs(U(:,1:cur),U(:,cur+1:next));
    R(1:next,cur+1:next) = tmp;
    B(cur+1:next,cur+1:next) = tmp(cur+1:next,:);
    
    [ub,sb,vb] = svd(R(1:next,1:next));
    s = diag(sb);

    if noexpand
        rnorm = sqrt(diag(ub(end-BlS+1:end,:)'*RtR*ub(end-BlS+1:end,:)));
    end
    
    fprintf('Time: %f\t Iter: %d\t Matvecs: %d\t k: %d\t BlS: %d\t FroNorm: %d\t Multiples: %d\n',toc,i,mvs,next,BlS,norm(s)/af,multiples);
    hist{i} = {toc, i, mvs, next, BlS, s, rnorm, norm(s)/af, multiples};
    
    switch mode
        case 1  %Frobenius Norm with residual tolerance
            if norm(s(rnorm./s(1) < tol))/af > delta
                num_2_return = find(rnorm./s(1) < tol,1,'last');
                break;
            end
        case 2  %Residual Tolerance
            if next > k && all(rnorm(1:k) < s(1)*tol)
                num_2_return = k;
                break;
            end
        case 3 %Find all singular values above a global gap (May be too strict on residuals)
            index = find(s < alpha*s(1),1);
            if noexpand && s(index)+rnorm(index) < alpha*s(1) %&& ...
                    %max(rnorm(1:ceil(index/BlS)*BlS)) < 0.1*rnorm(end)
                %Perhaps we can only check residuals if a relaxed
                %multiplicity count shows potential problems?...
                num_2_return = index - 1;
                break;
            end
        case 4 %Find local gap
            gaps = 1 - (s(k+1:end)+rnorm(k+1:end))./s(1:end-k);
            index = find(gaps > alpha);
            if ~isempty(index)
                num_2_return = index + k; %? not sure about this
                break;
            end
    end
    
    %Always check, iters, MaxBasis, MaxMV, and MaxTime
    if i == iters || next >= MaxBasis || mvs + 2*BlS > MaxMV
        if mode < 5
            m1 = 'Failed to meet criteria within given number of iterations or reached maximum basis';
            m2 = 'Increase maximum basis or iterations if returned answers are not sufficient';
            warning('%s\n%s',m1,m2);
        end
        num_2_return = next;
        break;
    end
    if toc > MaxTime
        num_2_return = next;
        break;
    end
    
    prev = cur;
    cur = next;
    next = next + BlS;
end

U = U(:,1:next)*ub(:,1:num_2_return);
S = sb(1:num_2_return,1:num_2_return);
V = V(:,1:next)*vb(:,1:num_2_return);
fprintf('Time: %f\t Iter: %d\t Matvecs: %d\t k: %d\t BlS: %d\t FroNorm: %d\t Multiples: %d\n',toc,i,mvs,next,BlS,norm(s)/af,multiples);
hist{i} = {toc, i, mvs, cur, BlS, s, rnorm, norm(s)/af, multiples};
end

function [MV,MB,I] = setComputationLength(MV,MB,I,BlS)
config = 0;
if I ~= inf
config = config + 1;
end
if MV ~= inf
    config = config+2;
end
if MB ~= inf
    config = config+4;
end

%In general, priority goes from MB -> I -> MV
switch config
    case 0
        I = 20; MV = 2*BlS*I; MB =  I*BlS;
    case 1
        MV = 2*BlS*I; MB =  I*BlS;
    case 2
        I = ceil(MV/(2*BlS)); MB = I*BlS;
    case 4
        I = ceil(MB/BlS); MV = 2*BlS*I;
    case 3 %I takes priority over MV
        MV = 2*BlS*I; MB =  I*BlS;
    case 5 %MB takes priority over I (until restarting is added)
        I = ceil(MB/BlS);   %Delete with restarting
        MV = 2*BlS*I;
    case 6 %MB takes priority over MV (until restarting is added)
        MV = 2*MB;          %Delete with restarting
        I = ceil(MV/(2*BlS));
    case 7 %MB takes priority over all, iterations takes over MV
        I = ceil(MB/BlS);   %Delete with restarting
        MV = 2*BlS*I;
end
end
