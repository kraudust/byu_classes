clear all
close all
clc

load state_meas_data.mat

%Initialize map
N = 100;
map = zeros(N,N); %start everything with a probability of 0.5 (log probability of 0 since 0.5/(1-0.5) = 1 and log(1) = 0)

for i = 1:N^2
    
end
