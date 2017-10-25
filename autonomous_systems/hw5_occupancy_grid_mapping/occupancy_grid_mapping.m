clear all
close all
clc

load state_meas_data.mat

%Initialize map
N = 100;
l = zeros(N,N); %start everything with a probability of 0.5 (log probability of 0 since 0.5/(1-0.5) = 1 and log(1) = 0)
l0 = 0;

for k = 1:size(X,2)
    for i = 1:N
        for j = 1:N
            mi = [i;j];
            l(i,j) = l(i,j) + inverse_range_sensor_model(mi,X(:,k),z(:,:,k)) - l0;
        end
    end
end

% Convert l to actual probabilites from log probabilites
l_prob = 1 - 1./(1+exp(l));
bar3(l_prob);

