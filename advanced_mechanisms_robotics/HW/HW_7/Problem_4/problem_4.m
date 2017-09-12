clear all
clc

% Make sure that you source the two startup files for the robotics toolbox
% and for the machine vision toolbox

% run ~/Desktop/vision-3.4/startup_rvc.m
% run ~/Desktop/rvctools/startup_rvc.m

%define the robotics toolbox Puma 560 arm
mdl_puma560;

%set the Coulomb friction terms to zero to help with numerical simulation
p560 = p560.nofriction;

%define desired robot pose
q_des = pi/180*[45,45,-135,0,-90,0];
Tcdes_0 = p560.fkine(q_des); %camera is at end effector
Tp_0 = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];

% Define object position in base frame
obj_pos = Tcdes_0(1:3,4);
obj_pos(3) = 0;
Tp_0(1:3,4) = obj_pos;
obj =    [...
    -0.1000   -0.1000    0.1000    0.1000   -0.1000   -0.1000    0.1000    0.1000;...
    -0.1000    0.1000    0.1000   -0.1000   -0.1000    0.1000    0.1000   -0.1000;...
    -0.1000   -0.1000   -0.1000   -0.1000    0.1000    0.1000    0.1000    0.1000];
obj = obj*0.5;
obj(1,:) = obj(1,:) + obj_pos(1);
obj(2,:) = obj(2,:) + obj_pos(2);
obj(3,:) = obj(3,:) + obj_pos(3);

scatter3(obj(1,:),obj(2,:), obj(3,:))
p560.plot(q_des)

cam = CentralCamera('default');
cam = cam.move(Tcdes_0);
pixels = cam.project(obj);
%cam.plot(obj);
%cam.plot(pixels);

Tp_c = cam.estpose(obj, pixels)

Tcdes_p = inv(Tp_0)*Tcdes_0;
inv(Tcdes_p)
cam.plot_camera
%P = [0.3, 0.4, 3.0]';
%cam.project(P);

%cam.project(P, 'Tcam', transl(-0.5, 0, 0));