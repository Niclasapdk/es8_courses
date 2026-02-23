function [p,chi2]= xcorrtk(x,y,m,c)
% xcorrtk(x,y,m) plots the cross correlation cor(x(t),y(t+k)) vs. k for k=
% -m:m, y, m and clim are optinal
% 
% x: data 1
% y: data 2 (default x)
% m: lags included (default 24)
% c: confidence interval (default 0.95)

% 2004-04-02 Torben Knudsen
% Time-stamp: <2005-02-16 14:14:53 tku>
% Vestas Wind Systems A/S,  Alsvej 21, DK-8900 Randers, Denmark
% E-mail: tku@vestas.com

if nargin<4; c=[]; end;
if nargin<3; m=[]; end;
if nargin<2; y=[]; end;

if isempty(c); c= 0.95; end;
if isempty(m); m= 24; end;
if isempty(y);
  auto= 1;
else;
  auto= 0;
end;

clim= norminv(1-(1-c)/2);

if auto == 1;
  n= length(x);
  rho= real(xcorr(detrend(x,0),'coeff'));
  rho= rho(:);
  rho= rho(n:n+m);
  g= clim/sqrt(n)*ones(size(rho));
  plot(0:m,rho,'-',0:m,rho,'x',0:m,g,'--',0:m,-g,'--');
  chi2= rho(2:m+1)'*rho(2:m+1)*n;
  p= 1-chi2cdf(chi2,m);
  grid;
  xlabel('lag');
  title(['Correlation function, nobs= ' int2str(n) ', p= ' num2str(p)]);
else;
  n= size(x,1);
  rho= real(xcorr(detrend(x,0),detrend(y,0),'coeff'));
  rho= rho(n-m:n+m);
  g= clim/sqrt(n)*ones(size(rho));
  plot(-m:m,rho,'-',-m:m,rho,'x',-m:m,g,'--',-m:m,-g,'--');
  grid;
  xlabel('lag');
  title(['Crosscorrelation function, nobs= ' int2str(n)]);
end;
