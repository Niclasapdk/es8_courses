%NLSim simulates the system
% x(i)= a*sin(x(i-1)+phif)+b*u(i-1)+w(i-1);
% y(i)= sin(c*x(i)+phih)+v(i);
% 
% External input: None

% Time-stamp: <2022-09-15 08:23:04 tk>
% Version 1: 
% Version 2: 2016-10-19 08:11:33 tk Added comments and minor changes
% Torben Knudsen
% Aalborg University, Dept. of Electronic Systems, Section of Automation
% and Control
% E-mail: tk@es.aau.dk

%% Parameters
n= 100; 
% $$$ n= 10000; 
% Weekly non linear system
a= 0.95; k= 1; b= k*(1-a); c= 1; phif= 0; phih= 0; fu= 0.02;
sigmaw= 0.05*sqrt(1-a^2); sigmav= 0.1; sigmax0= sigmaw; x0= 0;

% Misc test of other parameters
if 0; k= 10; b= k*(1-a); end
if 0; phif= pi/16; phih= 0; end
if 1; c= 10; end
if 0; sigmav= 1; sigmax0= 1; end
if 0; sigmav= 0.01; end

% More non linear system
% $$$ phif= pi/16;
% $$$ phih= pi/16;
% $$$ c= 10;
% $$$ sigmaw= 0.5*sqrt(1-a^2); sigmav= 0.1; sigmax0= sigmaw; x0= 0;

%% Algorithm

w= sigmaw*randn(n,1);
v= sigmav*randn(n,1);
u= square((1:n)*fu*2*pi)';

x= zeros(n,1);
y= zeros(n,1);
x(1)= sigmax0*randn+x0;
y(1)= sin(c*x(1)+phih)+v(1);
for i= 2:n;
  x(i)= a*sin(x(i-1)+phif)+b*u(i-1)+w(i-1);
  y(i)= sin(c*x(i)+phih)+v(i);
end;

Res= mean([u x y]);
Res= [Res; std([u x y])];
disp(array2table(Res,'RowNames',{'Mean','Std'},...
                 'VariableNames',{'u' 'x' 'y'}))

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
