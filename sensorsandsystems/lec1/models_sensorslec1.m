%% Sensors & Systems - Model simulation
clear; clc; close all;

s = tf('s');

%% 1) Define models
% G1(s)
G1 = (0.5*s + 1) / (s^3 - 3*s^2 + 0.4*s + 2);

% G2(s): 2 outputs, 1 input (transfer vector)
G2 = [ (0.2*s^2 + 2*s + 1)/(s^2 + 4*s + 1);
       (0.2*s^2 + 1)/(s^2 + 4*s + 1) ];

% G3: state-space
A3 = [-4 -1;
       1  0];
B3 = [2; 0];
C3 = [ 0.6  0.4;
      -0.4  0.4];
D3 = [0; 0];

G3 = ss(A3,B3,C3,D3);

%% 2) Simulate responses
t = 0:0.01:20;

figure; step(G1,t); grid on; title('G1 step response');
figure; step(G2,t); grid on; title('G2 step response (2 outputs)');
figure; step(G3,t); grid on; title('G3 step response (2 outputs)');

% Optional: arbitrary input with lsim
u = sin(0.8*t) + 0.5*(t >= 5);
figure; lsim(G1,u,t); grid on; title('G1 lsim response');
figure; lsim(G2,u,t); grid on; title('G2 lsim response');
figure; lsim(G3,u,t); grid on; title('G3 lsim response');

%% 3) Stability + observability checks
% For transfer functions, convert to minimal SS first
sys1 = minreal(ss(G1));
[A1,~,C1,~] = ssdata(sys1);

sys2 = minreal(ss(G2));
[A2,~,C2,~] = ssdata(sys2);

% G3 already in SS
[A3s,~,C3s,~] = ssdata(G3);

isObs1 = rank(obsv(A1,C1)) == size(A1,1);
isObs2 = rank(obsv(A2,C2)) == size(A2,1);
isObs3 = rank(obsv(A3s,C3s)) == size(A3s,1);

isStable1 = all(real(pole(G1)) < 0);
isStable2 = all(real(pole(G2)) < 0);
isStable3 = all(real(eig(A3s)) < 0);

disp('--- Results ---');
fprintf('G1: observable = %d, stable = %d\n', isObs1, isStable1);
fprintf('G2: observable = %d, stable = %d\n', isObs2, isStable2);
fprintf('G3: observable = %d, stable = %d\n', isObs3, isStable3);

disp('Poles / Eigenvalues:');
disp('G1 poles:'), disp(pole(G1).');
disp('G2 poles:'), disp(pole(G2).');
disp('G3 eigenvalues:'), disp(eig(A3s).');
