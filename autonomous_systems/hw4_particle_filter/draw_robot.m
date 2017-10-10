function draw_robot(t,xt,m,r,phi,mu, particles)
    % process inputs to function
    x = xt(1,:);
    y = xt(2,:);
    theta = xt(3,:);
    
    % Parameters
    P = defineCircle;
    
    %define persistent variables
    persistent vehicle_handle
    persistent vehicle_path_handle
    persistent landmark_handle
    persistent measurement_handle
    persistent est_path_handle
    persistent particles_handle
    
    %create an axis
    axis([-10 10 -10 10])
    
    % first time function is called, initialize plot and persistent vars
    for i = 1:length(t)
        particles_i = reshape(particles(i,:,:),3,size(particles,3));
        if t(i)==0
            xlabel('X')
            ylabel('Y')
            [vehicle_handle, vehicle_path_handle, landmark_handle,measurement_handle,est_path_handle, particles_handle] = update_plot(x(i),y(i),theta(i),P,[],[],[],[],[], [],  x,y, i,m,r,phi,mu, particles_i);
        % at every other time step, redraw 
        else 
            update_plot(x(i),y(i),theta(i),P,vehicle_handle, vehicle_path_handle,landmark_handle,measurement_handle,est_path_handle, particles_handle, x,y, i,m,r,phi,mu, particles_i);
        end
        %pause(0.02)
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v_handle,p_handle,l_handle,m_handle,est_handle, part_handle] = update_plot(xi,yi,theta,P,v_handle,p_handle,l_handle,m_handle,est_handle,part_handle,x,y,i,m,r,phi,mu, part)
    R = [cos(theta) -sin(theta);...
        sin(theta) cos(theta)];
    pts = R * P.pts;
    pts(1,:) = pts(1,:) + xi;
    pts(2,:) = pts(2,:) + yi;

  if isempty(v_handle)
    v_handle = patch(pts(1,:),pts(2,:),'red');
    hold on
    p_handle = plot(xi,yi);
    l_handle = scatter(m(:,1),m(:,2),80,'filled','k');
    m_handle = scatter(xi + r(:,i).*cos(theta + phi(:,i)),yi + r(:,i).*sin(theta + phi(:,i)),'bx');
    est_handle = plot(xi,yi,'r');
    part_handle = scatter(part(1,:), part(2,:), 7, 'g');
  else
    set(v_handle,'XData',pts(1,:),'YData',pts(2,:));
    set(p_handle,'XData',x(1:i), 'YData',y(1:i));
    set(m_handle,'XData',(xi + r(:,i).*cos(theta + phi(:,i)))','YData',(yi + r(:,i).*sin(theta + phi(:,i)))');
    set(est_handle,'XData',mu(1,1:i),'yData',mu(2,1:i));
    set(part_handle, 'XData',part(1,:), 'YData',part(2,:));
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
