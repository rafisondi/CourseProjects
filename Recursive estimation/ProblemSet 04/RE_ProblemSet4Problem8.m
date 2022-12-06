% Extended Kalman Filter design for the Coupled Logistic Maps
%
% Course: Recursive Estimation, Spring 2013
% Problem Set: Kalman Filtering
%
% --
% ETH Zurich
% Institute for Dynamic Systems and Control
% Philipp Reist
% reistp@ethz.ch
% 2013
%
% --
% Revision history
% [20.05.13, PR]    First version
%

function [] = RE_ProblemSet4Problem8()

% Choose which part of the problem you would like to run:
part = 'a'; % 'a' or 'b'

% System parameters:
mu = 3.9;       % growth rate
a = 0.6;        % process noise
b = 0.15;       % measurement noise
if('a' == part)
    alpha = 0.07;   % migration parameter
else
    alpha = rand*0.1;   % draw random migration parameter
end

%% Initialize actual system and estimator:
% Initialize actual system: Monkey populations
s0 = rand(2,1); % draw random populations

% Estimator parameters:
Q = eye(2)*a^2/12; % process noise variance, identical for v1 and v2
R = eye(2)*b^2/12; % measurement noise variance, identical for w1, w2

% Simulation setup:
Nsteps = 200;   % number of steps to simulate
sk = zeros(2,Nsteps + 1); % array to store actual pops. for reference
sk(:,1) = s0;  % Copy initial populations

% Initialize estimator:
if('a' == part)
    sm0 = 0.5*ones(2,1);    % mean of random initial populations
    Pm0 = eye(2) * 1/12;    % variance of random initial populations
    smk = zeros(2,Nsteps + 1); % array to store posterior means
    Pm = zeros(2,2,Nsteps + 1);    % array to store posterior variances
else
    sm0 = [0.5*ones(2,1);0.05];    % mean of random initial populations and random alpha
    Pm0 = diag(1/12*[ones(1,2),0.01]); % variance of random initial populations
    smk = zeros(3,Nsteps + 1);     % array to store posterior means
    Pm = zeros(3,3,Nsteps + 1);   % array to store posterior variances
end

% copy initial values to storage arrays:
smk(:,1) = sm0;           % copy initial posterior
Pm(:,:,1) = Pm0;          % copy initial state variance

%% Simulate
for i = 2:(Nsteps+1)
    % Update simulation state:
    sk(:,i) = dynamics(sk(:,i-1));
    % Generate measurement
    z = getMeasurement(sk(:,i));
    % Estimator Update:
    % Prior:
    [spk,Ppk] = priorUpdate(smk(:,i-1),Pm(:,:,i-1));
    % Posterior:
    [smk(:,i),Pm(:,:,i)] = posteriorUpdate(spk, Ppk, z);
end

%% Plot results
if('a' == part)
    figure(1)
    subplot(2,1,1)
    plot(0:Nsteps,sk(1,:)),hold on
    plot(0:Nsteps,smk(1,:),'g'), hold off
    xlabel('k')
    ylabel('x')
    legend('Act. Pop.', 'Est. Pop.')
    subplot(2,1,2)
    plot(0:Nsteps,sk(2,:)),hold on
    plot(0:Nsteps,smk(2,:),'g'), hold off
    xlabel('k')
    ylabel('y')
    legend('Act. Pop.', 'Est. Pop.')
else
    figure(1)
    subplot(3,1,1)
    plot(0:Nsteps,sk(1,:)),hold on
    plot(0:Nsteps,smk(1,:),'g'), hold off
    xlabel('k')
    ylabel('x')
    legend('Act. Pop.', 'Est. Pop.')
    subplot(3,1,2)
    plot(0:Nsteps,sk(2,:)),hold on
    plot(0:Nsteps,smk(2,:),'g'), hold off
    xlabel('k')
    ylabel('y')
    legend('Act. Pop.', 'Est. Pop.')
    subplot(3,1,3)
    plot(0:Nsteps,ones(1,Nsteps+1)*alpha),hold on
    plot(0:Nsteps,smk(3,:),'g'), hold off
    xlabel('k')
    ylabel('\alpha')
    legend('Act. Mig. Par.', 'Est. Mig. Par.')
end

%% Nested functions
% dynamics: Update actual populations
% input:    skm, the populations at year k-1
% output:   skp,  the populations at year k
    function [skp]  = dynamics(skm)
        lm = mu*skm.*(1 - skm); % logistic map update
        skp(1,1) = (1-alpha)*lm(1) + (1-a*rand)*alpha*lm(2); % pops. update
        skp(2,1) = (1-alpha)*lm(2) + (1-a*rand)*alpha*lm(1);
    end

% getMeasurement: Obtain measurement from real system
% input:    s, state at time k
% output:   z, measurement at time k, corrupted by meas. noise
    function z = getMeasurement(s)
        z = (1-b*rand(2,1)).*s;
    end

% priorUpdate: EKF prior update
% input:    skm, posterior mean at time k-1
%           Pmm, posterior variance at time k-1
% output:    sp, prior mean at time k
%            Pp, prior variance at time k
    function [sp,Pp] = priorUpdate(skm, Pmm)
        % mean update:
        lm = mu*skm(1:2,1).*(1 - skm(1:2,1)); % logistic map update
        if('a' == part)
            sp(1,1) = (1-alpha)*lm(1) + (1-a/2)*alpha*lm(2); % mean update
            sp(2,1) = (1-alpha)*lm(2) + (1-a/2)*alpha*lm(1); % notice expected value of proc. noise
        else
            sp(1,1) = (1-skm(3))*lm(1) + (1-a/2)*skm(3)*lm(2); % mean update
            sp(2,1) = (1-skm(3))*lm(2) + (1-a/2)*skm(3)*lm(1); % notice expected value of proc. noise
            sp(3,1) = skm(3);
        end
        % Variance update:
        % Linearize the nonlinear dynamics first
        % State dyn: partial f / partial s = A(k-1) =: Akm
        dlmds = mu*(1-2*skm(1:2,1));   % partials of logistic map
        if('a' == part)
            Akm = [(1-alpha)*dlmds(1), (1-a/2)*alpha*dlmds(2);
                (1-a/2)*alpha*dlmds(1),(1-alpha)*dlmds(2)];
            % Noise dyn: partial f / partial v = L(k-1) =: Lkm
            Lkm = [0, -alpha*lm(2); -alpha*lm(1),0];
        else
            Akm = [(1-skm(3))*dlmds(1), (1-a/2)*skm(3)*dlmds(2), -lm(1) + (1-a/2)*lm(2);
                   (1-a/2)*skm(3)*dlmds(1), (1-skm(3))*dlmds(2), -lm(2) + (1-a/2)*lm(1);
                0, 0, 1];
            % Noise dyn: partial f / partial v = L(k-1) =: Lkm
            Lkm = [-skm(3)*[0, lm(2); lm(1),0];zeros(1,2)];
        end
        
        % variance update
        Pp = Akm*Pmm*Akm' + Lkm*Q*Lkm';
    end

% posteriorUpdate: EKF prior update
% input:     sp, prior mean at time k
%            Pp, prior variance at time k
%             z, measurement at time k
% output:    sm, posterior mean at time k
%            Pm, posterior variance at time k
    function [sm,Pm] = posteriorUpdate(sp, Pp, z)
        % Linearize measurement equations:
        if('a' == part)
            % partial z / partial s = H(k) =: Hk
            Hk = eye(2)*(1-b/2); % note the exp. value E[w] of the meas. noise
        else
            % partial z / partial s = H(k) =: Hk
            Hk = [eye(2)*(1-b/2),zeros(2,1)]; % note the exp. value E[w] of the meas. noise
        end
        % partial z / partial w = M(k) =: Mk
        Mk = -diag(sp(1:2));
        % Kalman gain:
        Kk = Pp*Hk'/(Hk*Pp*Hk' + Mk*R*Mk');
        % Mean update:
        sm = sp + Kk*(z - (1-b/2)*sp(1:2,1)); % note again E[w]
        % Variance update:
        Pm = (eye(length(sp)) - Kk*Hk)*Pp*(eye(length(sp)) - Kk*Hk)' + Kk*Mk*R*Mk'*Kk';
    end
end