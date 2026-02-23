%LSim simulates the system
% x(i)= a*x(i-1)+b*u(i-1)+w(i-1);
% y(i)= c*x(i)+v(i);
% 
% External input: None

% Time-stamp: <2022-09-14 10:35:48 tk>
% Version 1: 2015
% Version 2: 2016-10-19 08:11:33 tk Added comments and minor changes
% Torben Knudsen
% Aalborg University, Dept. of Electronic Systems, Section of Automation
% and Control
% E-mail: tk@es.aau.dk

%% Parameters

n= 100; 
a= 0.95; k= 1; b= k*(1-a); c= 1; fu= 0.02; x0= 0;
sigmaw= 0.05*sqrt(1-a^2); sigmav= 0.1; sigmax0= sigmaw; 
NormalP= 1; NormalM= 1;                 % Distribution of process and
                                        % measurement noise
phif= 0; phih= 0;                       % Only used in EKF.m

% Misc test of other parameters
if 0; sigmav= 1; end
if 0; sigmax0= 1; end
if 1; sigmav= 0.01; end

%% Algorithm

if NormalP;                             % Using normal (0,1) distribution
  w= sigmaw*randn(n,1);
else;                               % Using uniform mu= 0 Var=1 distribution
  w= sigmaw*(rand(n,1)-1/2)/sqrt(1/12);
end;
if NormalM;                             % Using normal (0,1) distribution
  v= sigmav*randn(n,1);
else;                               % Using uniform mu= 0 Var=1 distribution
  v= sigmav*(rand(n,1)-1/2)/sqrt(1/12);
end;
u= square((1:n)*fu*2*pi);
x= zeros(n,1);
y= zeros(n,1);
x(1)= sigmax0*randn+x0;
y(1)= c*x(1)+v(1);
for i= 2:n;
  x(i)= a*x(i-1)+b*u(i-1)+w(i-1);
  y(i)= c*x(i)+v(i);
end;

figure(1)
subplot(311)
plot(u);
title('u')
subplot(312)
plot(x)
title('x')
subplot(313)
plot(y)
title('y')
