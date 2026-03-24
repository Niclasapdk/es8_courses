function [exectime, data] = send_queue_task(seg, dataL)

global msgQueue

data = 0;
switch seg
    case 1
        exectime = 0.02;
        if ~isempty(msgQueue)
            % Pop the first message from the queue
            msg = msgQueue(1);
            msgQueue(1) = [];
        
            % Send it to CAN (example: node 1 -> node 2)
            priority = 0;
            ttSendMsg([1 2], msg, 250, priority);
        end
        
        % Schedule next periodic send
        T_period = 0.1;  % e.g., send every 1 second
        ttCreateJob('send_queue_task', ttCurrentTime + T_period);
    case 2
        exectime = -1;
end
