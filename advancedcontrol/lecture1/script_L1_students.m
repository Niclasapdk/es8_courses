% Tutorial on subspace system identification - Lecture 1
% A small program by Szymon Gres, AAU - Spring 2026
clear all;close all;clc; %WATCH OUT FOR CLEAR! 

set(0,'defaultTextInterpreter','latex');
set(0, 'defaultAxesTickLabelInterpreter','latex'); set(0, 'defaultLegendInterpreter','latex');

addpath([cd,'\utility_codes\'])
systemmatrices = importdata('systemmatrices.mat');

response_type = 'impulse response';

%% input variables that might be changed during the lecture

N = 100000; %number of samples to simulate

A = systemmatrices.A;
B = systemmatrices.B;
C = systemmatrices.C;
D = systemmatrices.D;

r = size(C,1); % number of outputs
u = size(B,2); % number of inputs 

fs = systemmatrices.fs;
dt = 1/fs;

sys = ss(A,B,C,D); %building a system object (Attention!: it requires the Control System toolbox https://ch.mathworks.com/products/control.html)

%% convert the continuous-time system to discrete-time (check c2d() )

sysd = []; % a dummy discrete-time system 

sysd = c2d(sys, dt, 'zoh'); % convert the continuous-time system to discrete-time using zero-order hold


%% check observability

% Check observability
O = obsv(ss(sysd).A, ss(sysd).C); % Compute the observability matrix
rankO = rank(O); % Determine the rank of the observability matrix
if rankO < r
    disp('The system is not observable.');
else
    disp('The system is observable.');
end


%% check controllability

C = ctrb(ss(sysd).A, ss(sysd).B);

rankC = rank(C);
if rankC < r
    disp('The system is not controllable.');
else
    disp('The system is controllable.');
end


%% simulate the system response: 
t = 0:dt:(N-1)*dt; %time vector for simulation
y = impulse(sysd,t); %impulse response (Attention!: it requires the Control System toolbox https://ch.mathworks.com/products/control.html)

%% subspace system identification 

n = 20; %model order; can be a scalar, or a vector of model orders in case we analyze stabilization diagram
p = 30; %number of block rows to compute the subspace matrix; Note!: (p+1)r > n, so the necessary condition is that p should be at least n/r - 1. The sufficient (theoretical) condition is that p>n 

if (isempty(n) || isempty(p))
    disp('Chose the model order and the number of time lags!')
    return
end

param.mthd = response_type;
param.p = p;
param.n = n;
param.dt = dt;
param.r = r;
param.u = u;

hk=SSImat(y,param); %compute the block Hankel matrix

idres = HoKalmanId(hk,param); %identification code

Dest = [];
switch response_type %in case we investigate impulse responses, D can bo obtained from the initial observations
    case 'impulse response'
        Dest = y(1,:)*dt; %multiplication with dt related to the impulse excitation in matlab
end

%% validate the obtained system matrices

Aest = idres.Aest;
Cest = idres.Cest;
Best = idres.Best;
Dest = Dest.'
sys_identified = ss(Aest, Best, Cest, Dest, dt);

%% Validation: Compare impulse responses
t_val = 0:dt:min(2, (N-1)*dt);  % Validation time (up to 2 seconds or available data)

% Generate impulse responses
y_original = impulse(sysd, t_val);
y_identified = impulse(sys_identified, t_val);

% Plot comparison for all outputs
figure('Position', [100, 100, 1200, 800]);
num_outputs = size(y_original, 2);

for i = 1:num_outputs
    subplot(ceil(num_outputs/2), 2, i);
    plot(t_val, y_original(:, i), 'b-', 'LineWidth', 2);
    hold on;
    plot(t_val, y_identified(:, i), 'r--', 'LineWidth', 1.5);
    xlabel('Time [s]');
    ylabel(['Output ', num2str(i)]);
    legend('Original', 'Identified', 'Location', 'best');
    title(['Output ', num2str(i), ' - Impulse Response']);
    grid on;
end
sgtitle(['System Identification Validation (n = ', num2str(n), ')']);

%% Compute fit metrics
% Normalized fit percentage (VAF - Variance Accounted For)
fit_percent = zeros(1, num_outputs);
for i = 1:num_outputs
    fit_percent(i) = 100 * (1 - norm(y_original(:,i) - y_identified(:,i)) / norm(y_original(:,i) - mean(y_original(:,i))));
end

fprintf('\n=== Validation Results ===\n');
fprintf('Fit percentages (VAF) for each output:\n');
for i = 1:num_outputs
    fprintf('  Output %d: %.2f%%\n', i, fit_percent(i));
end
fprintf('Average fit: %.2f%%\n', mean(fit_percent));
fprintf('========================\n\n');

%% Plot error
figure('Position', [100, 100, 1200, 800]);
for i = 1:num_outputs
    subplot(ceil(num_outputs/2), 2, i);
    error_signal = y_original(:, i) - y_identified(:, i);
    plot(t_val, error_signal, 'k-', 'LineWidth', 1);
    hold on;
    plot(t_val, zeros(size(t_val)), 'r--');
    xlabel('Time [s]');
    ylabel(['Error ', num2str(i)]);
    title(['Output ', num2str(i), ' - Identification Error']);
    grid on;
    
    % Add RMS error
    rms_error = sqrt(mean(error_signal.^2));
    text(0.7*max(t_val), max(error_signal)*0.8, ...
         sprintf('RMS: %.2e', rms_error), 'FontSize', 10);
end
sgtitle(['Identification Error (n = ', num2str(n), ')']);

%% Frequency domain comparison (Bode plot)
if r == 1 && u == 1  % Single input, single output
    figure('Position', [100, 100, 1000, 600]);
    bode(sysd, sys_identified, logspace(-2, log10(pi/dt), 100));
    legend('Original', 'Identified', 'Location', 'best');
    title(['Frequency Response Comparison (n = ', num2str(n), ')']);
    grid on;
else  % Multiple inputs/outputs - plot first I/O pair
    figure('Position', [100, 100, 1000, 600]);
    bode(sysd(1,1), sys_identified(1,1), logspace(-2, log10(pi/dt), 100));
    legend('Original', 'Identified', 'Location', 'best');
    title(['Frequency Response Comparison - Output 1, Input 1 (n = ', num2str(n), ')']);
    grid on;
end

%% Pole-Zero comparison
figure('Position', [100, 100, 1000, 500]);

subplot(1,2,1);
pzmap(sysd);
title('Original System - Pole-Zero Map');
grid on;

subplot(1,2,2);
pzmap(sys_identified);
title(['Identified System (n = ', num2str(n), ') - Pole-Zero Map']);
grid on;

%% Compare eigenvalues visually
figure('Position', [100, 100, 800, 600]);
theta = linspace(0, 2*pi, 100);
plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1.5);
hold on;
grid on;
axis equal;

% Original system eigenvalues
eig_original = eig(sysd.A);
plot(real(eig_original), imag(eig_original), 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'MarkerFaceColor', 'b');

% Identified system eigenvalues
eig_identified = eig(Aest);
plot(real(eig_identified), imag(eig_identified), 'rx', 'MarkerSize', 12, 'LineWidth', 2);

xlabel('Real Part');
ylabel('Imaginary Part');
title(['Eigenvalue Comparison (n = ', num2str(n), ')']);
legend('Unit Circle', 'Original System', 'Identified System', 'Location', 'best');
xlim([-1.2, 1.2]);
ylim([-1.2, 1.2]);

%% Summary statistics
fprintf('\n=== System Comparison Summary ===\n');
fprintf('Original system order: %d\n', size(sysd.A, 1));
fprintf('Identified system order: %d\n', n);
fprintf('Number of outputs: %d\n', r);
fprintf('Number of inputs: %d\n', u);
fprintf('\nStability:\n');
fprintf('  Original: max|eig| = %.4f\n', max(abs(eig_original)));
fprintf('  Identified: max|eig| = %.4f\n', max(abs(eig_identified)));
fprintf('\nCondition numbers (from identification):\n');
fprintf('  Magnitude ratio: %.4f\n', idres.cond_magnitude);
fprintf('  Eigenvector cond: %.4f\n', idres.cond_eigenvectors);
fprintf('==================================\n\n');









