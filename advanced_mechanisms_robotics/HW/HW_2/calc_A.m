function [ A ] = calc_A( theta, d, a, alpha )
%CALC_A Summary of this function goes here
%   Calculate the symbolic A matrix from the dh parameters where A1 = transformation
%   from the 1 coordinate frame to the 0 coordinate frame. Any angles for alpha must
%   be in degrees. This is because sin(pi/2) does not = 0 while sind(90)
%   does. Theta should still be in radians.

A = [cos(theta) -sin(theta)*cosd(alpha)  sin(theta)*sind(alpha)   a*cos(theta);...
    sin(theta)  cos(theta)*cosd(alpha)   -cos(theta)*sind(alpha)  a*sin(theta);...
    0           sind(alpha)              cosd(alpha)              d;...
    0           0                       0                       1];

end

