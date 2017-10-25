clear all
close all
clc

load state_meas_data2.mat
% load mclain_data.mat
% load mclain_map.mat
% map(50,1) = 1;

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
            %mi = [i-1 + 0.5;j - 1 + 0.5];
            mi = [i,j];
            l(i,j) = l(i,j) + inverse_range_sensor_model(mi,X(:,k),z(:,:,k),thk) - l0;
        end
    end
    prob = 1 - 1./(1+exp(l));
    plot(X(1,k),X(2,k),'r.')
    hold on
    if k == 1
        plot_prob = surf(prob');
    else
        set(plot_prob,'ZData',prob');
    end
    pause(0.00001);
end

