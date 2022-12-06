%**************************************************************************
%   Model Predictive Engine Control                                       *
%   Spring 2021, IDSC, ETH Zurich                                         *
%   Problem Set 08, Exercise 1a: Single Shooting                          *
%**************************************************************************

%% Initialization
clear; clc; close all; path(pathdef);
if ispc == 1
    addpath('../../providedCode/casadi-matlabR2014b-v3.3.0');
elseif ismac == 1
    addpath('../../providedCode/casadi-matlabR2015a-v3.3.0');
else
    error('Unrecognized OS. Neither PC nor Mac!')
end % if
import casadi.*

%% Options
options.Ts        	= 0.5;          % Sampling time [s]
options.tFinal     	= 10;           % Final time [s]
options.x0       	= [0; 1];       % Initial condition of system
options.maxIter    	= 100;          % Maximum iterations for solver
options.nRK4        = 5;            % Number of RK4 intervals per time step

% Parse options
options.N = options.tFinal/options.Ts;  % Horizon length
time = (0:options.Ts:options.tFinal)';  % Time vector (for plotting)

%% Definition of model dynamics and objective function (continuous)
% States and inputs
x       = MX.sym('x',2) ;   % FILL IN
u       = MX.sym('u',1) ;   % FILL IN
nStates = length(x);
nInputs = length(u);

% System dynamics
x1dot   = (1 - x(2)^2)*x(1) - x(2) + u   % FILL IN
x2dot   = x(1)   % FILL IN
xdot    = [x1dot; x2dot];

% Objective function
J       =  x(1) + x(2) + u   % FILL IN

% Create CasADI functions
fxdot 	= Function('fxdot',{x,u},{xdot})   % FILL IN
fJ      = Function('fJ',{x,u},{J})   % FILL IN

%% Integration/discretization using RK4
xStart  = MX.sym('xStart',nStates,1);   % Initial condition of integration
u       = MX.sym('u',nInputs,1);        % Control input
TRK4    = options.Ts/options.nRK4;      % Step size of each RK4 interval /// Intermediate integration values

% Loop over intervals
xEnd    = xStart;                       % Initialization
JEnd    = 0;                            % Initialization
for l = 1:options.nRK4  % Given: u(1d) and a state x_k(2d) we calculate to 
                        % next point x_k+1 using four integration points 
    %Integrate 4 points xdot 
    k1 = fxdot(xEnd , u);
    k2 = fxdot(xEnd + TRK4/2 * k1, u);
    k3 = fxdot(xEnd + TRK4/2 * k2, u);
    k4 = fxdot(xEnd + TRK4/2 * k3, u);
    xEnd = xEnd + TRK4/6*(k1+2*k2+2*k3+k4); %%% NEW STATE k+1
     %Integrate Cost function
    JEnd = JEnd + TRK4 * fJ(x,u);       % COST of K+1 in dependence of x_k and u i.e l_k
end % for

% Create CasADi functions

fxDisc  = Function('fxDisc',{xStart,u},{xEnd});         
fJDisc  = Function('fJDisc',{xStart,u},{xEnd});             

%% Construct NLP
% Initialization
optVars   	= [];   % Vector of optimization variables (i.e. states and inputs)
optVars0    = [];   % Initial guess for optimization variables
lb          = [];   % Lower bound of optimization variables
ub          = [];   % Upper bound of optimization variables
Jk          =  0;   % Initialization of objective function
g           = [];   % (In-)equality constraints
lbg         = [];   % Lower bound on (in-)equality constraints
ubg         = [];   % Upper bound on (in-)equality constraints

% Pre-define CasADi variables
U = MX.sym('U_',nInputs,1,options.N);
Xk = options.x0; % Initial state vector
% Construct NLP step-by-step
for k = 1:options.N
    % Integration of system dynamics and objective function
    xEnd = fxDisc( xStart , U(k));         	% FILL IN
    %Jk = fJDisc(Xk(k), U(k));  %local cost
    
    % Add (in-)equality constraints (in this case state constraints)
    g   = [-Xk(1)-0.25];      % FILL IN
    lbg = [lbg;     ];      % FILL IN
    ubg = [ubg;     ];      % FILL IN
    
    % Add inputs to vector of optimization variables
    optVars = [U   	];          % FILL IN
    
    % Lower- and upper bounds for inputs
    lb = [-1    ];        % FILL IN
    ub = [1     ];        % FILL IN
    
    % Add initial guess for inputs
    optVars0 = [0 ,0       ];   	% FILL IN
    
    % Overwrite state for next iteration
    Xk  = Xk + xEnd           	% FILL IN
    
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
uOpt = [optVarsOpt; NaN];

% Simulate system to get states
xOpt = options.x0;
for k = 1:options.N
    % FILL IN
end
x1Opt = xOpt(1,:);
x2Opt = xOpt(2,:);

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