function [exectime, data] = disturbance_code(seg,dataL)

global msgQueue

networkNbr = 1;  % select a random network (1-3)
msg = [];                        % empty message
priority = 0;                    % highest priority
lambda=30;
u = 0.1 + 0.9*rand();  % between 0.1 and 1s                   
%msg = [ttCurrentTime]; 
msg.ttime = ttCurrentTime;
msg.payload = rand();  % optional
T=1;
data=1;

switch seg
  case 1
    exectime = 0.005;
    % Add message to the queue instead of sending
    if isempty(msgQueue)
        msgQueue = msg;
    else
        msgQueue(end+1) = msg;
    end

    ttCreateJob('generator_task',ttCurrentTime+u);
  case 2
     exectime = -1;
end


