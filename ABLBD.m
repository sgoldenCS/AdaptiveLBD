function [U,S,V] = ABLBD_Driver(A,varargin)

p = inputParser;

defaultk = -1;
defaultBlock = 20;
defaultAdaptive = 1e-4;
defaultMatvecs = inf;
defaultTime = inf;
defaultBasis = inf;
defaultCriteria = 1;
defaultCritArgs = struct('delta',0.9);

checkMatrix = @(x) issparse(x) || ismatrix(x);
checkInt = @(x) mod(x,1) == 0;
checkBool = @(x) x == 0 || x == 1;

addRequired(p,'A',checkMatrix);
addOptional(p,'k',defaultk,checkInt);
addOptional(p,'BlockSize',defaultBlock,checkInt);
addParameter(p,'Adaptive',defaultAdaptive,checkBool);
addParameter(p,'MaxMV',defaultMatvecs,checkInt);
addParameter(p,'MaxTime',defaultTime,@(x) x > 0);
addParameter(p,'MaxBasis',defaultBasis,checkInt);
addParameter(p,'Criteria',defaultCriteria,checkInt);
addParameter(p,'CriteriaArgs',defaultCritArgs);

parse(p,A,varargin{:});

[U,S,V,hist] = ABLBD_Main(p.Results);
