function animate_CSM(t,phi,theta,psi,n)
%ANIMATE_CSM Displays an animation of the CSM 
%   INPUTS: t - time vector with a constant step size
%           phi - vector of roll angles in degrees
%           theta - vector of pitch angles in degrees
%           psi - vector of yaw angles in degrees
%           n - playback speed 
dt = t(2)-t(1);
num_in_sec = 1/dt;
figure()

for i = 1:n*round(num_in_sec/18.5,0):length(t)
    drawCSM([t(i), phi(i)*pi/180, theta(i)*pi/180, psi(i)*pi/180])
end
end

function drawCSM(u)
% 
%     % process inputs to function
    phi   = u(2);
    theta = u(3);
    psi   = u(4);
    t     = u(1);
%     t = linspace(0,1,100); phi = linspace(0,0,100); theta = linspace(0,pi/2,100); psi = linspace(0,0,100);
    
    % drawing parameters
    COM = 0.2;
    
    % define persistent variables 
    persistent CM_handle
    persistent SM_handle 
    persistent axes1_handle
    persistent axes2_handle
    persistent axes3_handle
    axis([-6 6 -6 6 -6 6])
    % first time function is called, initialize plot and persistent vars
    if t==0
        %figure(), clf
        view([-46.084 -77.037 51.962]);
        xlabel('X')
        ylabel('Y')
        zlabel('Z')
        hold on
        CM_handle = drawCM_(phi,theta,psi,COM, []);
        SM_handle = drawSM_(phi,theta,psi,COM, []);
        axes1_handle = drawAxes1_(phi,theta,psi,COM,[]);
        axes2_handle = drawAxes2_(phi,theta,psi,COM,[]);
        axes3_handle = drawAxes3_(phi,theta,psi,COM,[]);
        
         
    % at every other time step, redraw mass
    else 
        %hold on
        drawCM_(phi,theta,psi,COM, CM_handle);
        drawSM_(phi,theta,psi,COM, SM_handle);
        axes1_handle = drawAxes1_(phi,theta,psi,COM,axes1_handle);
        axes2_handle = drawAxes2_(phi,theta,psi,COM,axes2_handle);
        axes3_handle = drawAxes3_(phi,theta,psi,COM,axes3_handle);
    end
end


function handle = drawCM_(phi,th,psi,COM, handle)
    [x,y,z] = cylinder([3.9/2 0],100);
    theta = pi/2;
    R = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
    R321 = [cos(psi)*cos(th) sin(psi)*cos(th) -sin(th);...
            -sin(psi)*cos(phi)+cos(psi)*sin(th)*sin(phi) cos(psi)*cos(phi)+sin(psi)*sin(th)*sin(phi) cos(th)*sin(phi);...
            sin(psi)*sin(phi)+cos(psi)*sin(th)*cos(phi) -cos(psi)*sin(phi)+sin(psi)*sin(th)*cos(phi) cos(th)*cos(phi)];

    X = zeros(2,length(x));
    Y = zeros(2,length(x));
    Z = zeros(2,length(x));
    
    for i=1:length(x)
        M1 = R * [x(1,i) y(1,i) z(1,i)]';
        M2 = R * [x(2,i) y(2,i) z(2,i)]';
    
        X(1,i) = M1(1) + COM;
        Y(1,i) = M1(2);
        Z(1,i) = M1(3);
        X(2,i) = M2(1) + COM;
        Y(2,i) = M2(2);
        Z(2,i) = M2(3);
        
        S = R321 * [X(1,i);Y(1,i);Z(1,i)];
        X(1,i) = S(1); Y(1,i) = S(2); Z(1,i) = S(3);
        S = R321 * [X(2,i);Y(2,i);Z(2,i)];
        X(2,i) = S(1); Y(2,i) = S(2); Z(2,i) = S(3);
    end
    

  if isempty(handle)
    handle = surf(X,Y,Z);
    colormap([0 0 0]);
  else
    set(handle,'XData',X,'YData',Y,'ZData',Z);
    drawnow  %limitrate
  end

end

function handle = drawSM_(phi,th,psi,COM, handle)
    [x,y,z] = cylinder(3.9/2,100);
    l = 3.5;
    z(2,:) = l;
    theta = pi/2;
    R = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
    R321 = [cos(psi)*cos(th) sin(psi)*cos(th) -sin(th);...
            -sin(psi)*cos(phi)+cos(psi)*sin(th)*sin(phi) cos(psi)*cos(phi)+sin(psi)*sin(th)*sin(phi) cos(th)*sin(phi);...
            sin(psi)*sin(phi)+cos(psi)*sin(th)*cos(phi) -cos(psi)*sin(phi)+sin(psi)*sin(th)*cos(phi) cos(th)*cos(phi)];
        
    X = zeros(2,length(x));
    Y = zeros(2,length(x));
    Z = zeros(2,length(x));
    
    for i=1:length(x)
        M1 = R * [x(1,i) y(1,i) z(1,i)]';
        M2 = R * [x(2,i) y(2,i) z(2,i)]';
    
        X(1,i) = M1(1) - l + COM;
        Y(1,i) = M1(2);
        Z(1,i) = M1(3);
        X(2,i) = M2(1) - l + COM;
        Y(2,i) = M2(2);
        Z(2,i) = M2(3);
        
        S = R321 * [X(1,i);Y(1,i);Z(1,i)];
        X(1,i) = S(1); Y(1,i) = S(2); Z(1,i) = S(3);
        S = R321 * [X(2,i);Y(2,i);Z(2,i)];
        X(2,i) = S(1); Y(2,i) = S(2); Z(2,i) = S(3);
    end
    

  if isempty(handle)
    handle = surf(X,Y,Z);
    colormap([0.5 0.5 0.5]);
  else
    set(handle,'XData',X,'YData',Y,'ZData',Z);
    drawnow  %limitrate
  end

end

function handle = drawAxes1_(phi,th,psi,COM, handle)
    l = 5;
    R321 = [cos(psi)*cos(th) sin(psi)*cos(th) -sin(th);...
        -sin(psi)*cos(phi)+cos(psi)*sin(th)*sin(phi) cos(psi)*cos(phi)+sin(psi)*sin(th)*sin(phi) cos(th)*sin(phi);...
        sin(psi)*sin(phi)+cos(psi)*sin(th)*cos(phi) -cos(psi)*sin(phi)+sin(psi)*sin(th)*cos(phi) cos(th)*cos(phi)];
    
    x1 = [0 l];
    y1 = [0 0];
    z1 = [0 0];
    
    for i=1:2
        S = R321 * [x1(i);y1(i);z1(i)];
        x1(i) = S(1); y1(i) = S(2); z1(i) = S(3);
    end

  if isempty(handle)
    handle = plot3(x1,y1,z1,'b','LineWidth',4);
  else
    set(handle,'XData',x1,'YData',y1,'ZData',z1);
    drawnow  %limitrate
  end

end

function handle = drawAxes2_(phi,th,psi,COM, handle)
    l = 5;
    R321 = [cos(psi)*cos(th) sin(psi)*cos(th) -sin(th);...
        -sin(psi)*cos(phi)+cos(psi)*sin(th)*sin(phi) cos(psi)*cos(phi)+sin(psi)*sin(th)*sin(phi) cos(th)*sin(phi);...
        sin(psi)*sin(phi)+cos(psi)*sin(th)*cos(phi) -cos(psi)*sin(phi)+sin(psi)*sin(th)*cos(phi) cos(th)*cos(phi)];
    
    x2 = [0 0];
    y2 = [0 -l];
    z2 = [0 0];

    
    for i=1:2
        S = R321 * [x2(i);y2(i);z2(i)];
        x2(i) = S(1); y2(i) = S(2); z2(i) = S(3);
    end

  if isempty(handle)
    handle = plot3(x2,y2,z2,'g','LineWidth',4);
  else
    set(handle,'XData',x2,'YData',y2,'ZData',z2);
    drawnow  %limitrate
  end

end

function handle = drawAxes3_(phi,th,psi,COM, handle)
    l = 5;
    R321 = [cos(psi)*cos(th) sin(psi)*cos(th) -sin(th);...
        -sin(psi)*cos(phi)+cos(psi)*sin(th)*sin(phi) cos(psi)*cos(phi)+sin(psi)*sin(th)*sin(phi) cos(th)*sin(phi);...
        sin(psi)*sin(phi)+cos(psi)*sin(th)*cos(phi) -cos(psi)*sin(phi)+sin(psi)*sin(th)*cos(phi) cos(th)*cos(phi)];

    x3 = [0 0];
    y3 = [0 0];
    z3 = [0 -l];

    
    for i=1:2
        S = R321 * [x3(i);y3(i);z3(i)];
        x3(i) = S(1); y3(i) = S(2); z3(i) = S(3);

    end

  if isempty(handle)
    handle = plot3(x3,y3,z3,'r','LineWidth',4);
  else
    set(handle,'XData',x3,'YData',y3,'ZData',z3);
    drawnow  %limitrate
  end

end