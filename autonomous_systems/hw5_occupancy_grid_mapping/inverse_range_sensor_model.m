function l = inverse_range_sensor_model(mi, xt, zt, thk)
% Table 9.2 Algorithm in probabilistic robotics
xi = mi(1);
yi = mi(2);
x = xt(1);
y = xt(2);
th = xt(3);

%Sensor parameters
l0 = 0;
p_occ = 0.7;
p_free = 0.4;
l_occ = log(p_occ/(1-p_occ));
l_free = log(p_free/(1-p_free));
alpha = 1; %thickness of obstacles in m
beta = 5*pi/180; %the angle the sensor shoots out at
z_max = 150; %m

%Compute range and bearing to cell mi and the closest beam index 
r = sqrt((xi-x)^2 + (yi-y)^2);
phi = atan2(yi-y, xi-x) - th;
[~, k] = min(abs(phi - thk)); % find the sensor index
if isnan(zt(1,k))
    zt(1,k) = 9999999;
end
if r > min(z_max, zt(1,k) + alpha/2) || abs(phi - thk(k)) > beta/2
    l = l0;
elseif zt(1,k) < z_max && abs(r - zt(1,k)) < alpha/2
    l = l_occ;
elseif r <= zt(1,k)
    l = l_free;
end
end

