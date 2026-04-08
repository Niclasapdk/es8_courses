function disturbance_init1

global msgQueue
msgQueue = []; % intitialize queue
msgQueue = struct('ttime', {}, 'payload', {});

data=1;
% Initialize TrueTime kernel
ttInitKernel('prioFP');  % scheduling policy - fixed priority

% Random disturbance generator
ttCreateTask('generator_task', 1, 'disturbance_code', data);
ttCreateJob('generator_task', ttCurrentTime);

% Periodic sender from queue
ttCreateTask('send_queue_task', 1, 'send_queue_task', data);
ttCreateJob('send_queue_task', ttCurrentTime + 0.2);  % start slightly later


