%EKF runs the extended Kalman filters for the system in NLSim.m
% x(i)= a*sin(x(i-1)+phif)+b*u(i-1)+w(i-1);
% y(i)= sin(c*x(i)+phih)+v(i);
% 
% External input: Data and parameters from NLSim.m (or LSim.m)

% Time-stamp: <2024-02-06 10:17:02 tk>
% Version 1: 2015
% Version 2: 2016-10-19 06:42:51 tk Error in time update that caused non
%            white residuals corrected 
%            (yhm= sin(c*xhp+phih) -> yhm= sin(c*xhm+phih))
% Version 3: 2024-02-06 10:16:17 tk Minor changes
% Torben Knudsen
% Aalborg University, Dept. of Electronic Systems, Section of Automation
% and Control
% E-mail: tk@es.aau.dk

%% Algorithm

% Initialising
xh0= x0; P0= sigmax0^2; R= sigmav^2; Q= sigmaw^2;

XHM= zeros(1,n);
YHM= zeros(1,n);
XHP= zeros(1,n);
KPmPp= [];

xhm= xh0; 
yhm= sin(c*xhm+phih);
Pm= P0;

% EKF
% Notice that the lines marked with "Collecting output" is only needed for plotting
% etc. They are not a necessarry part of the filter
for i= 1:n;
  XHM(i)= xhm;                          % Collecting output
  YHM(i)= yhm;                          % Collecting output
  % Measurement update
  H= c*cos(c*xhm+phih);
  K= Pm*H'/(H*Pm*H'+R);
  xhp= xhm+K*(y(i)-yhm);
  XHP(i)= xhp;                          % Collecting output
  Pp= (1-K*H)*Pm*(1-K*H)'+K*R*K';
  KPmPp= [KPmPp; K Pm Pp];              % Collecting output
  % Time update
  xhm= a*sin(xhp+phif)+b*u(i);
  Phi= a*cos(xhp+phif);
  Pm= Phi*Pp*Phi'+Q;
  yhm= sin(c*xhm+phih);
end;
  
% Generating results
figure(2)
subplot(321)
plot([x XHM' XHP'])
title('x XHM XHP')
subplot(322)
plot(x-XHM')
title('x-XHM')
subplot(323)
plot(x-XHP')
title('x-XHP')
subplot(324)
plot(y-YHM')
title('y-YHM')
subplot(325)
XCorrtk(y-YHM');
Res= mean([y-YHM' x-XHM' x-XHP']);
Res= [Res; std([y-YHM' x-XHM' x-XHP'])];
Res= [Res; sqrt(diag([y-YHM' x-XHM' x-XHP']'*[y-YHM' x-XHM' x-XHP']/n))'];
disp(array2table(Res,'RowNames',{'Mean','Std','RMSE'},...
                 'VariableNames',{'ytm' 'xtm' 'xtp'}))
figure(3)
subplot(311)
plot(KPmPp(:,1))
title('K')
subplot(312)
plot(KPmPp(:,2))
title('Pm')
subplot(313)
plot(KPmPp(:,3))
title('Pp')
