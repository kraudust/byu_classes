function draw_robot(t,xt,lm,r,phi,mu, sigma)
    % Robot Circle Parameters
    P = defineCircle;
    
    %define persistent variables
    persistent h %structure containing all the handles
    
    %create an axis
    axis([-10 10 -10 10])
    
    % first time function is called, initialize plot and persistent vars
    for i = 1:length(t)
        if t(i)==0
            xlabel('X')
            ylabel('Y')
            h = update_plot(xt,P,h, i,lm,r,phi,mu, sigma);
            
        % at every other time step, redraw 
        else 
            update_plot(xt,P,h, i,lm,r,phi,mu, sigma);
        end
%         pause()
%         pause(0.02)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h_st = update_plot(xt,P,h_st,i,lm,r,phi,mu, sigma)
    % set to true if you want to plot the measurments
    plot_meas = false;
    
    % true robot points
    R = [cos(xt(3,i)) -sin(xt(3,i));...
        sin(xt(3,i)) cos(xt(3,i))];
    pts = R * P.pts;
    pts(1,:) = pts(1,:) + xt(1,i);
    pts(2,:) = pts(2,:) + xt(2,i);
    
    % estimated robot points
    R = [cos(mu(3,i)) -sin(mu(3,i));...
        sin(mu(3,i)) cos(mu(3,i))];
    pts2 = R * P.pts;
    pts2(1,:) = pts2(1,:) + mu(1,i);
    pts2(2,:) = pts2(2,:) + mu(2,i);
    
  if isempty(h_st)
    h_st.v_handle = patch(pts(1,:),pts(2,:),'red'); % true robot
    h_st.v_handle.FaceVertexAlphaData = 0.2;
    h_st.v_handle.FaceAlpha = 'flat';
    hold on
    h_st.est_v_handle = patch(pts2(1,:),pts(2,:),'blue'); % estimated robot
    h_st.est_v_handle.FaceVertexAlphaData = 0.2;
    h_st.est_v_handle.FaceAlpha = 'flat';
    h_st.path_handle = plot(xt(1,i),xt(2,i),'r'); % true path
    h_st.path_est_handle = plot(xt(1,i),xt(2,i),'b'); % estimated path
    h_st.lm_handle = scatter(lm(:,1),lm(:,2),20,'filled','k'); % true landmark position
    h_st.lm_est_handle = scatter(mu(4:2:end,i),mu(5:2:end,i),20,'filled','r'); %estimated landmark pos.
    ellipse_data = plot_covariance(mu(1:2,i), sigma(1:2,1:2,i));
    h_st.cov_ellipses(1) = plot(ellipse_data(:,1),ellipse_data(:,2), 'r');
    for j = 1:(size(mu, 1) - 3)/2
        ellipse_data = plot_covariance(mu(4+2*(j-1):5+2*(j-1), i), sigma(4+2*(j-1):5+2*(j-1), 4+2*(j-1):5+2*(j-1), i));
        h_st.cov_ellipses(j+1) = plot(ellipse_data(:,1),ellipse_data(:,2), 'r');
    end
    %h_st.cov_ellipses = plot(ellipse_data(:,1),ellipse_data(:,2));
    if plot_meas == true
        h_st.m_handle = scatter(xt(1,i) + r(:,i).*cos(xt(3,i) + phi(:,i)),xt(2,i) +...
            r(:,i).*sin(xt(3,i) + phi(:,i)),'bx'); % measurements
    end
    
  else
    set(h_st.v_handle,'XData',pts(1,:),'YData',pts(2,:)); %true robot
    set(h_st.est_v_handle,'XData',pts2(1,:),'YData',pts2(2,:)); % estimated robot
    set(h_st.path_handle,'XData',xt(1,1:i), 'YData',xt(2,1:i)); % true path
    set(h_st.path_est_handle,'XData',mu(1,1:i),'YData',mu(2,1:i)); % estimated path
    set(h_st.lm_est_handle,'XData',mu(4:2:end,i),'YData',mu(5:2:end,i)); % estimated landmark positions
    %ellipse_data = plot_covariance(mu(1:2,i), sigma(1:2,1:2,i));
    ellipse_data = plot_covariance(mu(1:2,i), sigma(1:2,1:2,i));
    set(h_st.cov_ellipses(1), 'XData',ellipse_data(:,1),'YData',ellipse_data(:,2));
    for j = 1:(size(mu, 1) - 3)/2
        ellipse_data = plot_covariance(mu(4+2*(j-1):5+2*(j-1), i), sigma(4+2*(j-1):5+2*(j-1), 4+2*(j-1):5+2*(j-1), i));
        set(h_st.cov_ellipses(j+1), 'XData',ellipse_data(:,1),'YData',ellipse_data(:,2));
        %h_st.cov_ellipses(j+1) = plot(ellipse_data(:,1),ellipse_data(:,2));
    end
    if plot_meas == true
        set(h_st.m_handle,'XData',(xt(1,i) + r(:,i).*cos(xt(3,i) + phi(:,i)))',...
            'YData',(xt(2,i) + r(:,i).*sin(xt(3,i) + phi(:,i)))'); % measurements
    end
    
    drawnow
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define the robot circle parameters
function P = defineCircle
    rad = 0.6; %radius
    th = 0:pi/50:2*pi;
    P.pts = [0,rad*cos(th);0,rad*sin(th)];
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
