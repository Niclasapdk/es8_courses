%% run_EKF_KF_exercises_clean.m
% Clean self-contained solution for the EKF/KF exercise set
% Nonlinear system:
%   x_k = a*sin(x_{k-1}+phif) + b*u_{k-1} + w_{k-1}
%   y_k = sin(c*x_k + phih) + v_k
%
% Compares:
%   - Linear KF (using nominal linear model around 0)
%   - EKF (using nonlinear model + Jacobians)
%
% The script runs the 4 exercise cases and prints RMSE summary.

clear; clc; close all;

%% ---------------- User settings (copy from NLSim.m if needed) ----------------
pBase.a      = 0.95;
kGain        = 1;
pBase.b      = kGain*(1 - pBase.a);

pBase.c      = 1;      % changed in case (d)
pBase.phif   = 0;      % changed in case (c)
pBase.phih   = 0;      % changed in case (b)

% Noise / initialization (replace with exact values from your NLSim.m if needed)
pBase.Q      = 0.02^2;   % process noise variance
pBase.R      = 0.05^2;   % measurement noise variance
pBase.x0hat  = 0;
pBase.P0     = 0.10^2;

N            = 300;      % number of samples per run
Nmc          = 200;      % Monte Carlo runs (increase to 1000+ for smoother stats)
doPlots      = true;     % plot one demo trajectory per case

% Input (square-wave-like) settings
uAmp         = 1.0;
uPeriod      = 40;       % samples per full cycle
rngBase      = 42;       % reproducibility

%% ---------------- Define exercise cases ----------------
cases = {
    struct('name','(a) Initial parameters',         'c',1,  'phif',0,      'phih',0)
    struct('name','(b) Measurement phase shift',    'c',1,  'phif',0,      'phih',pi/16)
    struct('name','(c) State phase shift',          'c',1,  'phif',pi/16,  'phih',0)
    struct('name','(d) Strong nonlinear meas. c=10','c',10, 'phif',0,      'phih',0)
};

%% ---------------- Run all cases ----------------
results = struct([]);

for ic = 1:numel(cases)
    p = pBase;
    p.c    = cases{ic}.c;
    p.phif = cases{ic}.phif;
    p.phih = cases{ic}.phih;

    rmseKF_mc  = zeros(Nmc,1);
    rmseEKF_mc = zeros(Nmc,1);

    % Store one example run for plotting
    demo = struct();

    for m = 1:Nmc
        rng(rngBase + 1000*ic + m); % same seed per MC run for both filters in this case

        % Simulate ONE dataset, then run both filters on the same data
        [xTrue, yMeas, u] = simulate_nonlinear_system(p, N, uAmp, uPeriod);

        [xKF,  yhatKF,  innovKF]  = run_linear_kf(yMeas, u, p);
        [xEKF, yhatEKF, innovEKF] = run_ekf(yMeas, u, p);

        % Posterior state RMSE (ignore k=1 if you want strict predict/update alignment)
        rmseKF_mc(m)  = sqrt(mean((xTrue(2:end) - xKF(2:end)).^2));
        rmseEKF_mc(m) = sqrt(mean((xTrue(2:end) - xEKF(2:end)).^2));

        if m == 1
            demo.xTrue   = xTrue;
            demo.yMeas   = yMeas;
            demo.u       = u;
            demo.xKF     = xKF;
            demo.xEKF    = xEKF;
            demo.yhatKF  = yhatKF;
            demo.yhatEKF = yhatEKF;
            demo.innovKF = innovKF;
            demo.innovEKF= innovEKF;
        end
    end

    results(ic).name         = cases{ic}.name;
    results(ic).p            = p;
    results(ic).rmseKF_mean  = mean(rmseKF_mc);
    results(ic).rmseKF_std   = std(rmseKF_mc);
    results(ic).rmseEKF_mean = mean(rmseEKF_mc);
    results(ic).rmseEKF_std  = std(rmseEKF_mc);
    results(ic).improvement  = 100*(1 - results(ic).rmseEKF_mean/results(ic).rmseKF_mean);
    results(ic).demo         = demo;

    if doPlots
        plot_case_demo(cases{ic}.name, demo, p);
    end
end

%% ---------------- Print summary table ----------------
fprintf('\n=== EKF vs KF summary (posterior state RMSE over %d Monte Carlo runs) ===\n', Nmc);
fprintf('%-34s | %-12s | %-12s | %-10s\n', 'Case', 'KF RMSE', 'EKF RMSE', 'EKF gain');
fprintf('%s\n', repmat('-',1,78));
for ic = 1:numel(results)
    fprintf('%-34s | %10.5f | %10.5f | %8.1f%%\n', ...
        results(ic).name, ...
        results(ic).rmseKF_mean, ...
        results(ic).rmseEKF_mean, ...
        results(ic).improvement);
end

%% ---------------- Local functions ----------------
function [x, y, u] = simulate_nonlinear_system(p, N, uAmp, uPeriod)
    % Generates square-wave-like input, true state, and nonlinear measurements.
    %
    % Model:
    %   x_k = a*sin(x_{k-1}+phif) + b*u_{k-1} + w_{k-1}
    %   y_k = sin(c*x_k + phih) + v_k

    x = zeros(N,1);
    y = zeros(N,1);
    u = zeros(N,1);

    % square-wave-like input without toolbox dependencies
    for k = 1:N
        s = sin(2*pi*(k-1)/uPeriod);
        if s >= 0
            u(k) = uAmp;
        else
            u(k) = -uAmp;
        end
    end

    % Initial state
    x(1) = p.x0hat + sqrt(p.P0)*randn;
    y(1) = sin(p.c*x(1) + p.phih) + sqrt(p.R)*randn;

    for k = 2:N
        w = sqrt(p.Q)*randn;
        v = sqrt(p.R)*randn;
        x(k) = p.a*sin(x(k-1) + p.phif) + p.b*u(k-1) + w;
        y(k) = sin(p.c*x(k) + p.phih) + v;
    end
end

function [xhat, yhat, innov] = run_linear_kf(y, u, p)
    % Linear KF using nominal linearization around x=0 and zero phases:
    %   x_k ≈ a*x_{k-1} + b*u_{k-1} + w
    %   y_k ≈ c*x_k + v
    %
    % Important: This is intentionally mismatched when phif/phih ~= 0 or c is strongly nonlinear.

    N = numel(y);
    xhat  = zeros(N,1);   % posterior estimate x_k^+
    Pplus = zeros(N,1);

    yhat  = nan(N,1);     % predicted measurement
    innov = nan(N,1);

    % Nominal linear model used by the KF
    A = p.a;
    B = p.b;
    C = p.c;
    Q = p.Q;
    R = p.R;

    % Init
    xhat(1)  = p.x0hat;
    Pplus(1) = p.P0;

    for k = 2:N
        % Time update
        xpred = A*xhat(k-1) + B*u(k-1);
        Ppred = A*Pplus(k-1)*A + Q;

        % Measurement prediction (linear)
        ypred = C*xpred;

        % Innovation
        nu = y(k) - ypred;
        S  = C*Ppred*C + R;
        K  = (Ppred*C)/S;

        % Measurement update
        xhat(k) = xpred + K*nu;

        % Joseph covariance update (scalar form)
        IminusKC = (1 - K*C);
        Pplus(k) = IminusKC*Ppred*IminusKC + K*R*K;

        yhat(k)  = ypred;
        innov(k) = nu;
    end
end

function [xhat, yhat, innov] = run_ekf(y, u, p)
    % EKF for the nonlinear scalar system:
    %   f(x,u) = a*sin(x+phif) + b*u
    %   h(x)   = sin(c*x + phih)
    %
    % Jacobians:
    %   F = a*cos(x+phif)
    %   H = c*cos(c*x+phih)

    N = numel(y);
    xhat  = zeros(N,1);   % posterior estimate x_k^+
    Pplus = zeros(N,1);

    yhat  = nan(N,1);     % predicted measurement h(x_k^-)
    innov = nan(N,1);

    % Init
    xhat(1)  = p.x0hat;
    Pplus(1) = p.P0;

    for k = 2:N
        % ---- Time update ----
        x_prev = xhat(k-1);

        xpred = p.a*sin(x_prev + p.phif) + p.b*u(k-1);
        Fk    = p.a*cos(x_prev + p.phif);
        Ppred = Fk*Pplus(k-1)*Fk + p.Q;

        % ---- Measurement update ----
        ypred = sin(p.c*xpred + p.phih);
        Hk    = p.c*cos(p.c*xpred + p.phih);

        nu = y(k) - ypred;
        S  = Hk*Ppred*Hk + p.R;
        K  = (Ppred*Hk)/S;

        xhat(k) = xpred + K*nu;

        % Joseph covariance update (scalar)
        IminusKH = (1 - K*Hk);
        Pplus(k) = IminusKH*Ppred*IminusKH + K*p.R*K;

        yhat(k)  = ypred;
        innov(k) = nu;
    end
end

function plot_case_demo(caseName, demo, p)
    % Plot one example trajectory for visual comparison.

    t = (1:numel(demo.xTrue)).';

    figure('Name', caseName, 'Color', 'w');
    tiledlayout(3,1, 'Padding','compact', 'TileSpacing','compact');

    nexttile;
    plot(t, demo.xTrue, 'LineWidth', 1.2); hold on;
    plot(t, demo.xKF,   '--', 'LineWidth', 1.0);
    plot(t, demo.xEKF,  '-.', 'LineWidth', 1.0);
    grid on;
    ylabel('State x_k');
    title(sprintf('%s   |   c=%.2g, \\phi_f=%.3f, \\phi_h=%.3f', caseName, p.c, p.phif, p.phih));
    legend('True state','KF','EKF','Location','best');

    nexttile;
    plot(t, demo.yMeas, '.', 'MarkerSize', 6); hold on;
    plot(t, demo.yhatKF,  '--', 'LineWidth', 1.0);
    plot(t, demo.yhatEKF, '-.', 'LineWidth', 1.0);
    grid on;
    ylabel('Measurement y_k');
    legend('Measured y_k','KF predicted y','EKF predicted y','Location','best');

    nexttile;
    plot(t, demo.innovKF,  '--', 'LineWidth', 1.0); hold on;
    plot(t, demo.innovEKF, '-.', 'LineWidth', 1.0);
    grid on;
    xlabel('Sample k');
    ylabel('Innovation');
    legend('KF innovation','EKF innovation','Location','best');
end