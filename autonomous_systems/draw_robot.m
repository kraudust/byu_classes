function draw_robot(u)
    % process inputs to function
    t = u(1);
    x = u(2);
    y = u(3);
    theta = u(4);
    
    % Parameters
    P = defineCircle;
    
    %define persistent variables
    persistent vehicle_handle

    axis([-10 10 -10 10])
    % first time function is called, initialize plot and persistent vars
    if t==0
        xlabel('X')
        ylabel('Y')
        vehicle_handle = update_plot(x,y,theta,P,[]);

    % at every other time step, redraw 
    else 
        update_plot(x,y,theta,P,vehicle_handle);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function handle = update_plot(x,y,theta,P,handle)
    R = [cos(theta) sin(theta);...
        -sin(theta) cos(theta)];
    pts = R * P.pts;
    pts(1,:) = pts(1,:) + x;
    pts(2,:) = pts(2,:) + y;

  if isempty(handle)
    handle = patch(pts(1,:),pts(2,:),'red');
    %handle = viscircles([1,1],2);
  else
    set(handle,'XData',pts(1,:),'YData',pts(2,:));
    drawnow
  end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the parameters
function P = defineCircle
    rad = 1; %radius
    th = 0:pi/50:2*pi;
    P.pts = [0,rad*cos(th);0,rad*sin(th)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
