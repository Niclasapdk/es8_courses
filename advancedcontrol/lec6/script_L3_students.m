% Tutorial on data-driven subspace system identification - Lecture 3
% A small program by Szymon Gres, AAU - Spring 2026
clear all;close all;clc; %WATCH OUT FOR CLEAR! 

set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex'); set(0, 'defaultLegendInterpreter','latex');
set(0,'defaultAxesFontSize',14)

addpath([cd,'\utility_codes\'])
systemmatrices = importdata('systemmatrices.mat');

dtype = 'io'; % io, oo

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
[y,~,x] = lsim(sysd,u,t); %random response (Attention!: it requires the Control System toolbox https://ch.mathworks.com/products/control.html) 

%% subspace system identification 

n = 12; %model order; can be a scalar, or a vector of model orders in case we analyze stabilization diagram
p = 35; %number of block rows to compute the subspace matrix; Note!: (p+1)r > n, so the necessary condition is that p should be at least n/r - 1. The sufficient (theoretical) condition is that p>n 

if (isempty(n) || isempty(p))
    disp('Chose the model order and the number of time lags!')
    return
end

param.dtype = dtype;
param.p = p;
param.n = n;
param.dt = dt;
param.r = r;
param.m = m;

res = subspaceId(y,u,param);





function res = subspaceId(Y,U,param)

    p = param.p;
    q = p+1;
    r = param.r;
    u = param.m;
    nmax = param.n;
        
    [Hdat,Htot] = data_Hankel(Y,U.',param);
    [proj,projf,yp1,yp0] = proj_Hankel(Hdat,param); % Exercises add covariance driven algorithm instead of N4SID we do not have the states from covariance algorithm
    
    % Data driven algorithm
    [U,S,~] = svd(proj,'econ');
    U1 = U(:,1:nmax);
    U2 = U(:,nmax+1:end);
    s1 = diag(S);
    s1 = s1(1:nmax);
    Ob  =  U1*diag(sqrt(s1));
    C  =  Ob(1:r,:);
    Ob_up  =  Ob(1:p*r,:);
    Ob_dn = Ob(r+1:(p+1)*r,:);
    A = pinv(Ob_up)*Ob_dn;
    %
    % Determine the matrices M and U2T
    R4 = Htot.YfUf;
    iR8 = inv(Htot.UfUf);
    %
    U2T = U2';
    M = U2T*(R4*iR8); % U2^T R_4 inv(R_8)

    % Determine the set of equations
    Mv = zeros(q*(r*q-nmax),u);
    L = zeros(q*(r*q-nmax),q*r);
    for k=1:q
      Mv((k-1)*(r*q-nmax)+1:k*(r*q-nmax),:) = M(:,(k-1)*u+1:k*u);
      L((k-1)*(r*q-nmax)+1:k*(r*q-nmax),1:(q-k+1)*r) = U2T(:,(k-1)*r+1:r*q);
    end
    Os = [eye(r),zeros(r,nmax);zeros(r*(q-1),r),Ob_up];
    Ls = L*Os;
    pinvLs = pinv(Ls);
    %
    % Solve least squares
    sol = pinvLs*Mv;

    % Extract the system matrices
    D = sol(1:r,:);
    B = sol(r+1:r+nmax,:);
    
    %% 
        
    % Kalman states;
    Xi  = pinv(Ob)  * proj;
    Xip = pinv(Ob_up) * projf;

    switch param.dtype 
        case 'io'
            Rhst = [       Xi ; yp0 ]; 	% Right hand side
            Lhst = [      Xip   ;  yp1]; % Left hand side
            sol_exact = [A B;C D];              

    end    

    resk = Lhst - sol_exact*Rhst; 			% Residuals

    cov_est = resk*resk';

    Qs = cov_est(1:nmax,1:nmax);Ss = cov_est(1:nmax,nmax+1:nmax+r);Rs = cov_est(nmax+1:nmax+r,nmax+1:nmax+r); 
    
    %% states 
    sig = dlyap(A,Qs);
    G = A*sig*C' + Ss;
    L0 = C*sig*C' + Rs;
    [K,~] = gl2kr(A,G,C,L0); 
    
    Xss=ltitr((A-K*C),K,Y).'; %steady state states   

    res.X_ss = Xss;
    res.A = A;
    res.B = B;
    res.C = C;
    res.D = D;

end


function [Hdat,Htot] = data_Hankel(Y,U,param)

    r = param.r;
    p = param.p;
    u = min(size(U));
    N = max(size(Y));
    
    q = p+1;
    pq = p+q;
    NN = N-pq;
    
    switch param.dtype
        case 'io'
            
            Uf = zeros((p+1)*u,NN);
            Up = zeros(q*u,NN);
            Yf = zeros((p+1)*r,NN);
            Yp = zeros(q*r,NN);

            for i=0:q-1
                Up(i*u + (1:u), :) = U(i+(1:NN),:)';
                Yp(i*r + (1:r), :) = Y(i+(1:NN),:)';
            end
            for i=0:p
                Uf(i*u + (1:u), :) = U(q+i+(1:NN),:)';
                Yf(i*r + (1:r), :) = Y(q+i+(1:NN),:)';
            end

            Hdat = [Up;Uf;Yp;Yf]./sqrt(NN); %   
            Htot.YfUf = Yf*Uf'./NN;
            Htot.UfUf = Uf*Uf'./NN;
          
    end
    

       
end    

function  [proj,projf,yp1,yp0] = proj_Hankel(Hdat,param)
    
    r = param.r;
    u = param.m;
    p = param.p;
    q = p + 1;
    n = param.n;
    
    switch param.dtype
        case 'oo'
            R = triu(qr(Hdat',0))'; %size of the R_full 2*(p+1)*r x N-pq 
            proj  = R((p+1)*r+1:end,1:(p+1)*r);% * R(1:(p+1)*n_c,1:(p+1)*n_c)'; %projection R21; size (p+1)*r x qr
            projf = R((p+1+1)*r+1 : end ,1:(p+1+1)*(r) );% * R(1:(p+1+1)*n_c,1:(p+1+1)*n_c)';
            
            yp0 = zeros(n,r);
            yp1 = R((p+1)*r+1:(p+1+1)*r,1:(p+1+1)*r);
        case 'io'

            R = triu(qr(Hdat',0))'; 
            R = R(1:2*q*(u+r),1:2*q*(u+r)); 	% Truncate
            
            mi2 = 2*u*q;
            Rf = R((2*u+r)*q+1:2*(u+r)*q,:); 	% Future outputs
            Rp = [R(1:u*q,:);R(2*u*q+1:(2*u+r)*q,:)]; % Past (inputs and) outputs
            Ru  = R(u*q+1:mi2,1:mi2); 		% Future inputs
            % Perpendicular Future outputs 
            Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(u+r)*q)]; 
            % Perpendicular Past
            Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(u+r)*q)]; 
            
            proj  = (Rfp*pinv(Rpp')')*Rp;
            
            Rf = R((2*u+r)*q+r+1:2*(u+r)*q,:); 	% Future outputs
            Rp = [R(1:u*(q+1),:);R(2*u*q+1:(2*u+r)*q+r,:)]; % Past (inputs and) outputs
            Ru  = R(u*q+u+1:2*u*q,1:mi2); 		% Future inputs
            % Perpendicular Future outputs 
            Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(u+r)*q)]; 
            % Perpendicular Past
            Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(u+r)*q)]; 
            
            projf  = (Rfp*pinv(Rpp')')*Rp; 		% Oblique projection
       
            yp0 = R(u*q+1:u*(q+1),:);
            yp1 = R((2*u+r)*q+1:(2*u+r)*q+r,:);
            
    end
    
end

