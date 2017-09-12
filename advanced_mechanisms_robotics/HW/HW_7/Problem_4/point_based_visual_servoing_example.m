run D:\Dropbox\rvctools_new\startup_rvc.m
run D:\Dropbox\vision-3.4\rvctools\startup_rvc.m

clear all;
clc;
close all;

% you will need to follow the instructions at the link below in order for
% this code to work.
% https://groups.google.com/forum/#!topic/robotics-tool-box/sy7c5YyMRvE

cam = CentralCamera('default');

%defining position of camera relative to ground.
Tc0 = transl(1,1,-3)*trotz(0.6);

%defining desired final camera pose relative to target object or part. It's
%easier to define and understand.
Tc_des_p = transl(0, 0, 1);

%Here we define the pbvs object in matlab
pbvs = PBVS(cam, 'T0', Tc0, 'Tf', Tc_des_p)

%define the step size that we'll take of the total desired camera
%transformation
pbvs.lambda = 0.1

%this will show the camera as the simulation runs.
figure();
pbvs.plot_camera();

%this runs the simulation, but only moves the camera a small amount, it is
%completely decoupled from a robot in this simulation
pbvs.run();

%when finished we can plot the final pixel movements and commanded
%end-effector velocity
figure();
pbvs.plot_p();
figure();
pbvs.plot_vel();