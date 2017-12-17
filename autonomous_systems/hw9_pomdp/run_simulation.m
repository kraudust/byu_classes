close all
clear all
clc
global meas_cor_prob
meas_cor_prob = 0.7;
% run value function
value_function
clear gamma i ind_no_dup indices j k k1 k2 N num_control_actions num_measurements pruned_indices state state_no_dup state_plot sum T tau u v val vprime yip_prime z
for i = 1:length(yip)
    s(i,:) = yip(i).state;
end

% find intersection points
k = 1;
for i = 1:size(s,1)-1
    for j = i:size(s,1)-1
        x(k) = (s(i,2) - s(j+1,2))/((s(j+1,1)-s(j+1,2))-(s(i,1)-s(i,2)));
        k = k+1;
    end
end

% locate import intersection points that change which control input to use (in the middle is u3)
plow = min(x) % anything less than this should result in action u1
phigh = max(x) % anything greater than this should result in action u2

figure()
plot(plotx,s)

% Run simulation with inital state of x1 and belief of 0.6
num_games = 1000;
for i = 1:num_games % run 10 simulations
    k = 1;
    bel(k) = 0.6; %belief
    x(k) = 1; %true state
    act = 3;
    r = 0; % reward
    while act == 3
        % first determine which action to choose based on belief
        if bel(k) < plow
            act = 1;
        elseif bel(k) > phigh
            act = 2;
        else
            act = 3;
        end

        % determine effect of action
        if act == 1 %action 1
            if x(k) == 1
                reward = -100
                r = r + reward;
            else
                reward = 100
                r = r + reward;
            end
        elseif act == 2 % action 2
            if x(k) == 1
                reward = 100
                r = r + reward;
            else
                reward = -50
                r = r + reward;
            end
        else % action 3
            r = r - 1;
            rand_var_act = rand();
            if rand_var_act > 0.8
                if x(k) == 1
                    x(k+1) = 1;
                else 
                    x(k+1) = 2;
                end
            else
                if x(k) == 1
                    x(k+1) = 2;
                else
                    x(k+1) = 1;
                end
            end
            % update belief bar
            bel_bar(k) = bel(k)*0.2 + (1-bel(k))*0.8;
            
            %take a measurement
            rand_var_meas = rand();
            if rand_var_meas > meas_cor_prob
                if x(k+1) == 1
                    z(k) = 2;
                else
                    z(k) = 1;
                end
            else
                if x(k+1) == 1
                    z(k) = 1;
                else
                    z(k) = 2;
                end
            end
            
            % update belief
            if z(k) == 1
                bel(k+1) = meas_cor_prob*bel_bar(k)/(meas_cor_prob*bel_bar(k) + (1-meas_cor_prob)*(1-bel_bar(k)));
            else
                bel(k+1) = (1-meas_cor_prob)*bel_bar(k)/((1-meas_cor_prob)*bel_bar(k) + meas_cor_prob*(1-bel_bar(k)));
            end
        end
        k = k + 1;
    end
    reward_sum(i) = r; 
end

% Determine win percentage
win_percent = sum(reward_sum > 0)/num_games



