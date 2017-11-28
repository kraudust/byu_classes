clear all
close all
clc

N = 100;

map = zeros(N,N);        % map dimension

% Initialize walls and obstacle maps as empty
walls = zeros(N,N);
obs1 = zeros(N,N);
obs2 = zeros(N,N);
obs3 = zeros(N,N);
goal = zeros(N,N);

% Define Rewards and MDP Parameters
wall_reward = -100;
obstacle_reward = -5000;
goal_reward = 100000;
global nom_cell_reward iter gamma prob
nom_cell_reward = -2;
discount_factor = 1;

% Create exterior walls
walls(1,1:N) = wall_reward;
walls(1:N,1) = wall_reward;
walls(N,1:N) = wall_reward;
walls(1:N,N) = wall_reward;

% Create single obstacle
obs1(20:40,30:80) = obstacle_reward;
obs1(10:20,60:65) = obstacle_reward;

% Another obstacle
obs2(45:65,10:45) = obstacle_reward;

% Another obstacle
obs3(43:92,75:85) = obstacle_reward;
obs3(70:80,50:75) = obstacle_reward;

% The goal states
goal(75:80,96:98) = goal_reward;

% Put walls and obstacles into map
map = walls + obs1 + obs2 + obs3 + goal;


gamma = 1;  % discount factor
iter = zeros(N,N);
prob = 0.8; % probability of moving in desired direction

% Plot map ----------------------------------------------------------------
% Sort through the cells to determine the x-y locations of occupied cells
[Mm,Nm] = size(map);
xm = [];
ym = [];
    for i = 1:Mm
        for j = 1:Nm
            if map(i,j)
                xm = [xm i];
                ym = [ym j];
            end
        end
    end

figure(1)
plot(xm,ym,'.')
axis([0 N+1 0 N+1]);
axis('square');
%--------------------------------------------------------------------------
% Make list of cells to iterate through that are not walls or obstacles or goal states
for i = 1:N
    for j = 1:N
        if map(i,j) == 0
            map(i,j) = nom_cell_reward;
            iter(i,j) = 1;
        end
    end
end

[Vhat, policy] = mdp_discrete_value_iteration(map);


% Make Plots --------------------------------------------------------------
% Plot Value Function
figure(2)
b = bar3(Vhat);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
axis([0 N+1 0 N+1]);
axis('square'); 
zlim([0.99 * goal_reward,goal_reward]); %can't really see it w/o this
view(-133,10)

% Plot Policy
figure(1);
xlim([0,100]);
ylim([0,100]);
hold on
for i = 1:2:N
    for j = 1:2:N
        if iter(i,j) == 1
            angle = (policy(i,j)-1) * pi/2;
            draw_arrow(i,j,1,angle)
        end
    end
end

% Plot Path
goal_states = [repmat([75:80],1,3)',[repmat(96,1,6)' ;repmat(97,1,6)' ;repmat(98,1,6)']];

xtraj = [28];
ytraj = [20];
ingoal = 0;
iters = 1;
while ingoal == 0
    angle = (policy(xtraj(iters),ytraj(iters)) - 1) * pi/2;
    xtraj = [xtraj xtraj(iters) - 1 * sin(angle)];
    ytraj = [ytraj ytraj(iters) + 1 * cos(angle)];
    iters = iters + 1;
    if ismember([xtraj(iters) ytraj(iters)],goal_states)
       ingoal = 1;
    end
end
plot(xtraj,ytraj,'r')



