function draw_robot(u)
    % process inputs to function
    t = u(1,:);
    x = u(2,:);
    y = u(3,:);
    theta = u(4,:);
    
    % Parameters
    P = defineCircle;
    
    %define persistent variables
    persistent vehicle_handle
    persistent vehicle_path
    
    %create an axis
    axis([-10 10 -10 10])
    
    % first time function is called, initialize plot and persistent vars
    for i = 1:length(t)
        if t(i)==0
            xlabel('X')
            ylabel('Y')
            [vehicle_handle, vehicle_path] = update_plot(x(i),y(i),theta(i),P,[],[], x,y, i);
        % at every other time step, redraw 
        else 
            update_plot(x(i),y(i),theta(i),P,vehicle_handle, vehicle_path, x,y, i);
        end     
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v_handle,p_handle] = update_plot(xi,yi,theta,P,v_handle,p_handle,x,y,i)
    R = [cos(theta) -sin(theta);...
        sin(theta) cos(theta)];
    pts = R * P.pts;
    pts(1,:) = pts(1,:) + xi;
    pts(2,:) = pts(2,:) + yi;

  if isempty(v_handle)
    v_handle = patch(pts(1,:),pts(2,:),'red');
    hold on
    p_handle = plot(xi,yi);
  else
    set(v_handle,'XData',pts(1,:),'YData',pts(2,:));
    set(p_handle,'XData',x(1:i), 'YData',y(1:i));
    drawnow
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the parameters
function P = defineCircle
    rad = 0.6; %radius
    th = 0:pi/50:2*pi;
    P.pts = [0,rad*cos(th);0,rad*sin(th)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
