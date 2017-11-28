clear all
close all
clc

N = 100;
Np = 100 + 2;

map = zeros(Np,Np);        % map dimension

% Initialize walls and obstacle maps as empty
walls = zeros(Np,Np);
obs1 = zeros(Np,Np);
obs2 = zeros(Np,Np);
obs3 = zeros(Np,Np);
goal = zeros(Np,Np);

% Define Rewards and MDP Parameters
wall_reward = -100;
obstacle_reward = -5000;
goal_reward = 100000;
nom_cell_reward = -2;
discount_factor = 1;

% Create exterior walls
walls(2,2:N) = wall_reward;
walls(2:N+1,2) = wall_reward;
walls(N+1,2:N+1) = wall_reward;
walls(2:N+1,N+1) = wall_reward;

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

global di gamma prob
gamma = 1;  % discount factor
di = []; % list of cells not to iterate over during value iteration
prob = 0.8; % probability of moving in desired direction

% Make list of cells to iterate through that are not walls or obstacles or goal states
for i = 1:Np
    for j = 1:Np
        if map(i,j) == 0
            map(i,j) = nom_cell_reward
        else
            di = [di [i;j]];
        end
    end
end



% Make Plots --------------------------------------------------------------
% Plot map
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
axis([0 Np+1 0 Np+1]);
axis('square');

figure(2)
b = bar3(map);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end
axis([0 Np+1 0 Np+1]);
axis('square'); 
view(-90,90)

