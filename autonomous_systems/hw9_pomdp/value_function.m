clear all
close all
clc
global meas_cor_prob
meas_cor_prob = 0.7;
yip(1).u = 0;
yip(1).state = [0,0];
T = 2; % horizon
N = 2; % number of possible states
num_control_actions = 3;
num_measurements = 2;
gamma = 1;
plotx = [1,0];
for tau = 1:T
    yip_prime.u = [];
    yip_prime.state = [];
    for k = 1:length(yip)
        for u = 1:num_control_actions
            for z = 1:2
                for j = 1:N
                    sum = 0;
                    for i = 1:N
                        sum = sum + yip(k).state(i)*calc_meas_prob(z,i)*calc_state_prob(i,u,j);
                    end
                    v(k,u,z,j) = sum;
                end
            end
        end
    end

    for u = 1:num_control_actions
        for k1 = 1:length(yip)
            for k2 = 1:length(yip)
                for i = 1:N
                    vprime(i) = gamma*(reward(i, u) + v(k1, u, 1, i) + v(k2, u, 2, i));
                end
                yip_prime.u = [yip_prime.u; u];
                yip_prime.state = [yip_prime.state; vprime];
            end
        end
    end
    for i = 1:length(yip_prime.u)
        yip(i).u = yip_prime.u(i);
        yip(i).state = yip_prime.state(i,:);
    end

    % Prune duplicates
    for i = 1:length(yip)
       state(i,:) = yip(i).state; 
    end
    [state_no_dup,ind_no_dup,~] = unique(state,'rows','stable');
    yip = yip(ind_no_dup);

    % Prune states that aren't the max in the value function
    val = zeros(size(state_no_dup,1),length(linspace(0,1,201)));
    k = 1;
    for i = linspace(0,1,201)
        for j = 1:size(state_no_dup,1)
            val(j,k) = interp1(plotx,state_no_dup(j,:),i);
        end
        k = k + 1;
    end
    [~, indices] = max(val,[],1);
    pruned_indices = unique(indices,'stable');

    yip = yip(pruned_indices);
    state = [];
end


for i = 1:length(yip)
   state_plot(i,:) = yip(i).state; 
end
% figure()
% plot(plotx,state_plot)





