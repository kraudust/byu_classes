run D:\Dropbox\rvctools_new\startup_rvc.m
run D:\Dropbox\vision-3.4\rvctools\startup_rvc.m 
clear all;
close all;
clc;


%define a default camera
cam = CentralCamera('default');

%define a 3D point in space and project the point into pixel coordinates
P = [1 1 5]';
p0 = cam.project(P)
cam.plot(p0)

%then plot it too
figure()
plot_sphere(P, 0.01, 'r');
hold on;
cam.plot_camera();
pause;

%now displace the camera a bit in the x-direction and look at pixel
%coordinates again, what does this tell us?
px = cam.project( P, 'Tcam', transl(0.1,0,0) )
pause;

%new position minus old divided by the distance moved approximates the
%derivative dp/dx - means that 1 m of camera motion in x-direction would
%give - 160 pixels of feature motion in the u-direction
( px - p0 ) / 0.1
pause;

%what about for z-axis translation? This gives equal motion in u and v
%directions
( cam.project( P, 'Tcam', transl(0, 0, 0.1) ) - p0 ) / 0.1
pause;

%what about for x-axis rotation? - causes most motion in v direction
( cam.project( P, 'Tcam', trotx(0.1) ) - p0 ) / 0.1
pause;

%MATLAB will calculate the image jacobian L(u,v,Z) for us
% first argument is the feature pixel coordinates
% second argument is the distance away of the feature in the z-direction 
% in meters
L = cam.visjac_p([672; 672], 5)
%each column shows us the effect on pixel coordinate's velocity due to the
%Cartesian velocity of the camera.
pause;

%this shows a flow field of velocity for a set of pixels in the image plane
%(i.e. we calculated an image Jacobian for each arrow shown). For moving in
%the x-direction, it looks like this:
fig_x_dir = figure()
cam.flowfield( [1 0 0 0 0 0] );
pause;

%for motion in the z-direction:
fig1=figure()
cam.flowfield( [0 0 1 0 0 0] );
pause;
close(fig1);

%for rotation about the z-direction;
fig1=figure()
cam.flowfield( [0 0 0 0 0 1] )
pause;
close(fig1);

%for rotation about y-axis is very similar to translation in x-axis with
%some curvature at outer points. Intution for this comes by moving head to
%right or translating head to right. In both cases, features move to the
%left. Also image Jacobian of a specific point we looked at above had very
%similar columns for x translation and y rotation.
fig_y_rot = figure()
cam.flowfield( [0 0 0 0 1 0] );
pause; 

%changing focal length from default of 8 mm to 20 mm instead, we see that
%the flow field becomes even more similar to x translation
fig_y_rot2 = figure()
cam.f = 20e-3;
cam.flowfield( [0 0 0 0 1 0] );