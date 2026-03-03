close all; clc; 

% Parameters
num_sensor = 8;
T = 1;
data_bits = 100;
O = 100;
N_data = 6;

% Arrival per packet:,
packet_size = N_data * data_bits + O;

% x,y, slope
y_offset = num_sensor*(data_bits*N_data+O);
slope = num_sensor*data_bits + (num_sensor*data_bits)/N_data;   

% Encode arrival constraint as rtccurve
arrival_sensor = rtccurve([0, y_offset, slope]);

% Encode service curve as rtccurve
service_curve = rtccurve([0, 0, 1000]);

% Plot maximum experienced delay (h) and backlog (v)
rtcplot(arrival_sensor, 'r', service_curve, 'g', 100);
h = rtcploth(arrival_sensor, service_curve) % Use this as appropiate tool to determine best N_data
v = rtcplotv(arrival_sensor, service_curve)


