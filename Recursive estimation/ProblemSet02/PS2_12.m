% Problem Set: Bayes Theorem - Problem 12
%
% *Object on Circle - Recursive Filtering Algorithm*
%
% Recursive Estimation
% Spring 2011
%
% --
% ETH Zurich
% Institute for Dynamic Systems and Control
% Angela Schöllig
% aschoellig@ethz.ch
%
% --
% Revision history
% [14.03.10, AS]    first version
%


clear

rand('state',0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configuration Constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Number of simulation steps
T = 100;

% Number of discrete steps around circle
N = 100;

% Actual probability of going CCW
PROB = 0.55;

% Model of probability of going CCW
PROB_MODEL = PROB;
%PROB_MODEL = 0.45;

% Location of distance sensor, as a multiple of the circle radius.  Can be
% less than 1 (inside circle), but must be positive (WLOG).
SENSE_LOC = 2;

% The sensor error is modeled as additive (a time of flight sensor, for
% example), uniformly distributed around the actual distance.  The units
% are in circle radii.  
ERR_SENSE = 0.50;

% Model of what the sensor error is
ERR_SENSE_MODEL = ERR_SENSE;
%ERR_SENSE_MODEL = 0.45;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% W(k,i) denotes the probability that the object is at location i at time
% k, given all measurements up to, and including, time k.  At time 0, this
% is initialized to 1/N, all positions are equally likely.
W = zeros(T+1,N);
W(0+1,:) = 1/N;

% The intermediate prediction weights, initialize here for completeness.
% We don't keep track of their time history.
predictW = zeros(1,N);

% The initial location of the object, an integer between 0 and N-1.
loc = zeros(T+1,1);
loc(0+1) = round(N/4);

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

for t = 1:T    
    %%%%%%%%%%%%%%%%%
    % Simulate System
    %%%%%%%%%%%%%%%%%
    
    % Process dynamics.  With probability PROB we move CCW, otherwise CW
    if (rand < PROB)
        loc(t+1) = mod(loc(t) + 1,N);
    else
        loc(t+1) = mod(loc(t) - 1,N);
    end
    
    % The physical location of the object is on the unit circle
    xLoc = cos(2*pi * loc(t+1)/N);
    yLoc = sin(2*pi * loc(t+1)/N);
        
    % Can calculate the actual distance to the object
    dist = sqrt( (SENSE_LOC - xLoc)^2 + yLoc^2);
    
    % Corrupt the distance by noise
    dist = dist + ERR_SENSE * 2 * (rand - 0.5);
    
    %%%%%%%%%%%%%%%%%%
    % Update Estimator
    %%%%%%%%%%%%%%%%%%
    
    % Prediction Step.  Here we form the intermediate weights which capture
    % the pdf at the current time, but not using the latest measurement.
    for i = 1:N
        predictW(i) = PROB_MODEL*W(t, 1+mod(i-2,N)) + (1-PROB_MODEL)*W(t, 1+ mod(i,N));
    end
    
    % Fuse prediction and measurement.  We simply scale the prediction step
    % weight by the conditional probability of the observed measurement
    % at that state.  We then normalize.
    for i = 1:N
        
        xLocHypo = cos(2*pi * (i-1)/N);
        yLocHypo = sin(2*pi * (i-1)/N);
        
        distHypo = sqrt( (SENSE_LOC - xLocHypo)^2 + yLocHypo^2);
        
        if abs(dist-distHypo) < ERR_SENSE_MODEL
            condProb = 1/(2*ERR_SENSE_MODEL);
        else
            condProb = 0;
        end
        
        W(t+1,i) = condProb * predictW(i);
        
    end
    
    % Normalize the weights.  If the normalization is zero, it means that
    % we received an inconsistent measurement.  We can either use the old
    % valid data, re-initialize our estimator, or crash. To be as
    % robust as possible, we simply re-initialize the estimator.
    normConst = sum(W(t+1,:));
    
    % Uncomment this line if we want to allow the program to crash.
    W(t+1,:) = W(t+1,:)/normConst;    normConst = 1.0;
    
    if (normConst > 1e-6)
        W(t+1,:) = W(t+1,:)/normConst;
    else
        W(t+1,:) = W(1,:);
    end
    
end
       
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
% Visualize the results
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

figure(1)
xVec = (0:N-1)/N;
yVec = 0:T;
mesh(xVec,yVec,W);
xlabel('POSITION x(k)/N ');
ylabel('TIME STEP k');
view([-30,40]);
hold on
% actual simulated position
plot3(loc/N,(0:T)',ones(T+1,1)*max(max(W)));
hold off 

findfigs