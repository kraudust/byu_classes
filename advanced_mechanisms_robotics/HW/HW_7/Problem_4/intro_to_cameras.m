% Make sure that you source the two startup files for the robotics toolbox
% and for the machine vision toolbox

run ~/Desktop/vision-3.4/startup_rvc.m
run ~/Desktop/rvctools/startup_rvc.m

%this makes a camera with a focal length of 0.015 m, but because it defines
%no pixel-to-meter relationship, any projecion will be the location of the
%the thing we are looking at in units of meters on the image plane
cam = CentralCamera('focal', 0.015);

%make a 3D point in the camera frame
P = [0.3, 0.4, 3.0]';

%we can project the point onto the image plane using the camera matrix,
%which is again in meters, not pixels for this example
cam.project(P)
pause;

%we can also do things like move the camera 0.5 m to the left
cam.project(P, 'Tcam', transl(-0.5, 0, 0) )
pause;


%Now we can define a camera with 10 micrometer square pixels, a 1280x1024
%resolution, and a principal point (the center of the image sensor) at
%(640, 512) in terms of pixels.
cam = CentralCamera('focal', 0.015, 'pixel', 10e-6, ...
'resolution', [1280 1024], 'centre', [640 512], 'name', 'mycamera')

%and we can still project the same point onto the image plane, but now
%we'll get pixel coordinates for that point that our camera is looking at.
cam.project(P)
pause;



%Next we can make a 3x3 grid of points in a plane in the real world. This
%function generates a grid in the xy-plane that is centred at the origin
%and moved 1.0 meter in the z-direction.
P = mkgrid(3, 0.2, 'T', transl(0, 0, 1.0));

%we can find points in camera image
cam.project(P)

%and now plot them too
cam.plot(P)
pause;


%now we can move the camera again and we'll see the perspective change
Tcam = transl(-1,0,0.5)*troty(0.9);

%and replot the new view
cam.plot(P, 'Tcam', Tcam)
pause;
cam.clf();

%we can make a 3D object instead with side length of 0.2 and centered at
%[0,0,1]
cube = mkcube(0.2, 'T', transl([0, 0, 1]) );

%now plot in the cube in the camera view:
cam.plot(cube)
pause;
cam.clf();

%can also make a mesh or "edge" representation which is possibly easier to
%visualize
[X,Y,Z] = mkcube(0.2, 'T', transl([0, 0, 1.0]), 'edge');

%and display it:
cam.mesh(X, Y, Z)
pause;
cam.clf();

%can again move camera and rotate a bit and then plot it
cam.T = transl(-1,0,0.5)*troty(0.8);
cam.mesh(X, Y, Z, 'Tcam', Tcam);
pause;
cam.clf();

%can show things like the object moving too
theta = [0:500]/100*2*pi;
[X,Y,Z] = mkcube(0.2, [], 'edge');
for th=theta
    T_cube = transl(0, 0, 1.5)*trotx(th)*troty(th*1.2)*trotz(th*1.3)
    cam.mesh( X, Y, Z, 'Tobj', T_cube );
    pause(0.01);
    cam.clf();
end
pause


%pose estimation - The pose estimation problem is to determine the pose of
%a target's coordinate frame with respect to the camera. We know the
%geometry of the object and have multiple points of the object in the
%object frame. Also assume we know the intrinsic parameters of the camera.
cam = CentralCamera('focal', 0.015, 'pixel', 10e-6, ...
'resolution', [1280 1024], 'centre', [640 512]);

%this makes a 3D object where we know local points in the object's frame
%(i.e. the vertices)
P = mkcube(0.2);

%Now imagine cube is at an arbirtrary and unkown pose with respect to the
%camera. We will define this pose here, but if all we have is the camera
%and can see the object, we won't know this transformation.
T_unknown = transl(0,0,0.5)*trotx(0.1)*troty(0.2)

%assume we can find the image points of known things like the vertices:
p = cam.project(P, 'Tobj', T_unknown);

%there exists an algorithm to find the unknown transformation which is just
%solving for the rotation and translation given corresponding real world
%coordinates and pixel coordinates (this uses the same math as the
%calibration we discussed in class).
T_est = cam.estpose(P, p)

% we can plot the points in the frame of reference of the object
figure();
plot_sphere(P, 0.01, 'r');
hold on;
trplot(eye(4,4), 'frame', 'T', 'color', 'b', 'length', 0.3);
hold on;

%can also plot camera - but need to invert transform if we want
%to say where camera is with respect to object
cam.plot_camera('Tcam', inv(T_est))

