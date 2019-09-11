function [U,S,V,hist] = ABLBD(A,varargin)
%% Modes:
% MaxBasis, MaxMV, MaxIter, and MaxTime are always monitored
% If nothing is set, MaxIter defaults to 20 iterations
%
% Currently, there are four modes that check additional criteria:
% 1) Frobenius norm with a residual tolerance
% 2) Residual Tolerance for k vectors
% 3) Find all values above a global gap
% 4) Find all values above a local gap
%
%
%
%
%
% The "Config" file allows for all options to be given using a *.m script

p = inputParser;

%% Defaults
defaultk = inf;
defaultBlock = 50;
defaultAdaptive = 1e-4;
defaultMatvecs = inf;
defaultTime = 60;
defaultBasis = inf;
defaultIter = inf;
defaultDelta = 0.9;
defaultAlpha = 0.01;
defaultTol = 1e-3;
defaultConfig = [];
defaultMode = 5;

%% Input checking functions
checkMatrix = @(x) issparse(x) || ismatrix(x);
checkInt = @(x) mod(x,1) == 0;
checkBool = @(x) x == 0 || x == 1;
checkConfig = @(file) ~ischar(file) && ~isStringScalar(file);

%% Parser Parameters
addRequired(p,'A',checkMatrix);
addOptional(p,'Mode',defaultMode,checkInt);
addParameter(p,'k',defaultk,checkInt);
addParameter(p,'BlockSize',defaultBlock,checkInt);
addParameter(p,'Adaptive',defaultAdaptive);
addParameter(p,'MaxMV',defaultMatvecs,checkInt);
addParameter(p,'MaxTime',defaultTime,@(x) x > 0);
addParameter(p,'MaxBasis',defaultBasis,checkInt);
addParameter(p,'MaxIter',defaultIter,checkInt);
addParameter(p,'Delta',defaultDelta);
addParameter(p,'Alpha',defaultAlpha);
addParameter(p,'tol',defaultTol);
addParameter(p,'Config',defaultConfig,checkConfig);


%% Call Main function
parse(p,A,varargin{:});
[U,S,V,hist] = ABLBD_Main(p.Results);
