clear all
close all
clc

load state_meas_data.mat
% load mclain_data.mat
% load mclain_map.mat
map(50,1) = 1;

%Initialize map
N = 100; %400 x 400 cells but dimension is 100 x 100 m
l = zeros(N,N); %start everything with a probability of 0.5 (log probability of 0 since 0.5/(1-0.5) = 1 and log(1) = 0)
l0 = 0;
%l0 = log(0.7/(1-0.7));
% l_prev = zeros(5*N,5*N);

for k = 1:size(X,2)
    for i = 1:N
%     for i = 0.125:0.25:N
        for j = 1:N
%         for j = 0.125:0.25:N
%             mi = [(i-1)*0.25 +0.125, (j-1)*0.25 +0.125];
            mi = [i-1 + 0.5;j - 1 + 0.5];
            l(i,j) = l(i,j) + inverse_range_sensor_model(mi,X(:,k),z(:,:,k),thk) - l0;
        end
    end
end

% Convert l to actual probabilites from log probabilites
l_prob = 1 - 1./(1+exp(l));
figure()
bar3(l_prob)
% figure()
% bar3(map)

