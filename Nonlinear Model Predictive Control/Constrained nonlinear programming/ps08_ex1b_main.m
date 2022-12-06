%**************************************************************************
%   Model Predictive Engine Control                                       *
%   Spring 2021, IDSC, ETH Zurich                                         *
%   Problem Set 08, Exercise 1b: Multiple Shooting                        *
%**************************************************************************

%% COPY THE FIRST PART OF YOUR CODE FROM Ex1a HERE
%  i.e. lines 7 to 65

%% Construct NLP
% Initialization
optVars     = [];   % Vector of optimization variables (i.e. states and inputs)
optVars0    = [];   % Initial guess for optimization variables
lb          = [];   % Lower bound of optimization variables
ub          = [];   % Upper bound of optimization variables
Jk          =  0;   % Initialization of objective function
g           = [];   % (In-)equality constraints
lbg         = [];   % Lower bound on (in-)equality constraints
ubg         = [];   % Upper bound on (in-)equality constraints

% Pre-define CasADi variables
U = MX.sym('U_',nInputs,1,options.N);
S = MX.sym('S_',nStates,1,options.N+1);

% Construct NLP step-by-step
for k = 1:options.N+1
    
    % System dynamics and objective function
    if k==options.N+1
        % Skip, because at final time step, no further integration of
        % system dynamics necessary.
    else
        % Integration of system dynamics and objective function
        if k==1
            % Hardcoded initial condition
            % FILL IN
            % FILL IN
        else
            % FILL IN
            % FILL IN
        end % if
        
        % Add equality constraint for continuity (i.e. closing gaps):
        % FILL IN
        % FILL IN
        % FILL IN
        
    end % if
    
    % States
    if k==1
        % Skip, because we have hardcoded the initial condition above
    else
        % Add states to vector of optimization variables
        % FILL IN
        
        % Lower- and upper bounds for states
     	% FILL IN
        % FILL IN
        
        % Add initial guess of states
        % FILL IN
        
    end % if
    
    % Inputs
    if k==options.N+1
        % Skip, because no control input at final time step
    else
        % Add inputs to vector of optimization variables
        % FILL IN
        
        % Lower- and upper bounds for inputs
        % FILL IN
        % FILL IN
        
        % Add initial guess for inputs
        % FILL IN
        
    end % if
    
end % for

%% Solve the NLP using IPOPT
% Create solver
optionsIPOPT = struct('ipopt',struct('max_iter',options.maxIter));
prob = struct('f',Jk,'x',optVars,'g',g);
solver = nlpsol('solver','ipopt',prob,optionsIPOPT);

% Solve
tic
sol = solver('x0',optVars0,'lbx',lb,'ubx',ub,'lbg',lbg,'ubg',ubg);
time_to_solve = toc

% Extract solution
optVarsOpt = full(sol.x);
uOpt = [optVarsOpt(1:3:end); NaN];
x1Opt = [options.x0(1); optVarsOpt(2:3:end)];
x2Opt = [options.x0(2); optVarsOpt(3:3:end)];

%% Plot
set(0,'defaulttextinterpreter','latex');
set(0,'defaultlegendinterpreter','latex');

fig1 = figure(1); clf;

ax(1) = subplot(2,1,1); hold on; box on; grid on;
plot(time,x1Opt,'b.','MarkerSize',18);
plot(time,x2Opt,'ro','MarkerSize',6,'LineWidth',1);
ylabel('$x$');
legend('$x_1$','$x_2$');

ax(2) = subplot(2,1,2); hold on; box on; grid on;
stairs(time,uOpt,'k','LineWidth',1);
ylabel('$u$');

xlabel('Time [s]');
linkaxes(ax,'x');

%% EOF