clc
clear all

load('hw7_prob3.mat');

% Make sure that you source the two startup files for the robotics toolbox
% and for the machine vision toolbox

% run ~/Desktop/vision-3.4/startup_rvc.m
% run ~/Desktop/rvctools/startup_rvc.m

cam = CentralCamera('default');
T1_est = cam.estpose(P1,p1)
T2_est = cam.estpose(P2,p2)
T3_est = cam.estpose(P3,p3)

cam2 = cam.move(inv(T1_est));

cam.plot(p1)
cam2.plot(P1)
