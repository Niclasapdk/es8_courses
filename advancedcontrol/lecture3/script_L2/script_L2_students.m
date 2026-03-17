% Tutorial on subspace system identification - Lecture 2
% A small program by Szymon Gres, AAU - Spring 2026
clear all;close all;clc; %WATCH OUT FOR CLEAR! 

set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex'); set(0, 'defaultLegendInterpreter','latex');

addpath([cd,'\utility_codes\'])
systemmatrices = importdata('systemmatrices.mat');

mthd = 'UPC'; % OOcov, UPC

%% input variables that might be changed during the lecture

N = 100000; %number of samples to simulate

A = systemmatrices.A;
B = systemmatrices.B;
C = systemmatrices.C;
D = systemmatrices.D;

r = size(C,1); % number of outputs
m = size(B,2); % number of inputs 

fs = systemmatrices.fs;
dt = 1/fs;

sys = ss(A,B,C,D); %building a system object (Attention!: it requires the Control System toolbox https://ch.mathworks.com/products/control.html)

%% convert the continuous-time system to discrete-time (check c2d() )

sysd = c2d(sys,dt);


%% simulate the system response: 
t = 0:dt:(N-1)*dt; %time vector for simulation
u = randn(m,N); %random white noise excitation 
y = lsim(sysd,u,t); %random response (Attention!: it requires the Control System toolbox https://ch.mathworks.com/products/control.html) 

%% subspace system identification 

n = 12; %model order; can be a scalar, or a vector of model orders in case we analyze stabilization diagram
p = 15; %number of block rows to compute the subspace matrix; Note!: (p+1)r > n, so the necessary condition is that p should be at least n/r - 1. The sufficient (theoretical) condition is that p>n 

if (isempty(n) || isempty(p))
    disp('Chose the model order and the number of time lags!')
    return
end

param.mthd = mthd;
param.p = p;
param.n = n;
param.dt = dt;
param.r = r;
param.m = m;

[hk,param]=SSImat(y,param); %compute the block Hankel matrix
res = subspaceId(hk,param);
Hdat = param.Hdat; % Det her er fra lec6 ses3 subsapce for at bygge LQ
% Now we need to find L21
R = triu(qr(Hdat',0))';
proj = R((p+1)*r+1:end,1:(p+1)*r);
projf = R((p+1+1)*r+1 : end , 1:(p+1+1)*(r)); % this for states calculation for kalman data matrice is shifted by 1
hk = proj;
[U, S, V] = svd(hk, 'econ');
S = diag(S);

%% validate the obtained system matrices

%Aest = idres.Aest;
%Cest = idres.Cest;
%Best = idres.Best;
 
% Truncate at model order n
U1 = U(:, 1:n);
S1 = diag(S(1:n));  % convert back to diagonal matrix

% Observability matrix
O = U1 * sqrtm(S1);

% C is the first r rows
C = O(1:r, :);

% A from the shift property
O_up = O(1:end-r, :);
O_down = O(r+1:end, :);
A = O_up \ O_down;

figure;
semilogy(S, 'o-');
xlabel('Index');
ylabel('Singular value');
title('Singular values');
grid on;
xline(n, 'r--', 'Chosen order');

