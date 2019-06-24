function [U,S,V,hist] = ABLBD(A,varargin)

p = inputParser;

defaultk = inf;
defaultBlock = 50;
defaultAdaptive = 1e-4;
defaultMatvecs = inf;
defaultTime = 60;
defaultBasis = inf;
defaultCriteria = 9; %Time and frobenius norm
defaultDelta = 0.9;
defaultAlpha = 0.01;
defaultTol = 1e-3;
defaultConfig = [];

checkMatrix = @(x) issparse(x) || ismatrix(x);
checkInt = @(x) mod(x,1) == 0;
checkBool = @(x) x == 0 || x == 1;
checkConfig = @(file) ~ischar(file) && ~isStringScalar(file);

addRequired(p,'A',checkMatrix);
addOptional(p,'k',defaultk,checkInt);
addParameter(p,'BlockSize',defaultBlock,checkInt);
addParameter(p,'Adaptive',defaultAdaptive,checkBool);
addParameter(p,'MaxMV',defaultMatvecs,checkInt);
addParameter(p,'MaxTime',defaultTime,@(x) x > 0);
addParameter(p,'MaxBasis',defaultBasis,checkInt);
addParameter(p,'Criteria',defaultCriteria,checkInt);
addParameter(p,'Delta',defaultDelta);
addParameter(p,'Alpha',defaultAlpha);
addParameter(p,'tol',defaultTol);
addParameter(p,'Config',defaultConfig,checkConfig);

parse(p,A,varargin{:});



[U,S,V,hist] = ABLBD_Main(p.Results);
