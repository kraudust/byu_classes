run D:\Dropbox\rvctools_new\startup_rvc.m
run D:\Dropbox\vision-3.4\rvctools\startup_rvc.m 
clear all;
close all;
clc;


%define a default camera
cam = CentralCamera('default');

%define real world points that are unknown to the control system
P = mkgrid( 2, 0.5, 'T', transl(0,0,3) );
cam.plot(P)
pause

%we can define the desired position of the target features in the image
%plane are a 400 x 400 square centered on the principal point of the camera
%image - bsxfun is just a matlab function, there are more straightforward
%ways to define this same set of image points
s_des = bsxfun(@plus, 200*[-1 -1 1 1; -1 1 1 -1], cam.pp');


%the camera is at an initial pose Tc and we can project the pixel
%coordinates into the image plane from there:
s = cam.project(P, 'Tcam', transl(0.9,0,0));
cam.plot(P, 'Tcam', transl(0.9,0,0));
pause;

%we can calculate an image jacobian which is an 8x6 matrix by assuming a
%depth for each feature
depth = 2; %this is in meters
L = cam.visjac_p(s, depth);

%we can calculate error
[r,c] = size(s)
s_reshape = reshape(s, [c*2, 1]);
s_des_reshape = reshape(s_des, [c*2, 1]);
e = s_des_reshape - s_reshape;

%use the control law we defined to find a desired camera twist (velocity
%and angular velocity)
lambda = 0.05;
v = pinv(L) * lambda*e
pause;


%this error calculation and control part could be put in a loop and we 
%could relate this camera velocity to joint velocities in order to command 
%specific motion. For this example, we can also just use the MATLAB IBVS
%object
Tc0 = transl(1,1,-3)*trotz(0.6);
ibvs = IBVS(cam, 'T0', Tc0, 'pstar', s_des);
ibvs.run()
figure()
ibvs.plot_p();
figure()
ibvs.plot_vel();
figure()
ibvs.plot_camera();
%ibvs.plot_jcond();
pause;
close all;

%what about the dependency on depth? We really don't know depth. We have
%two options: 1)there are methods to estimate depth as you move the camera.
%2)assume a depth which it turns out work pretty well.

%actual depth from Tc0 is 3 meters

%let depth be assumed to be 1 meter
ibvs = IBVS(cam, 'T0', Tc0, 'pstar', s_des, 'depth', 1)
ibvs.run(50)
figure()
ibvs.plot_p()
figure()
ibvs.plot_vel()
pause;

%let depth be assumed to be 10 meter
ibvs = IBVS(cam, 'T0', Tc0, 'pstar', s_des, 'depth', 10)
ibvs.run(50)
figure()
ibvs.plot_p()
figure()
ibvs.plot_vel()
pause;