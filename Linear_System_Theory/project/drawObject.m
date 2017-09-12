function drawObject(uu, a, b, c)
% x = [p, q,  r,  psi, th, phi, px,  py,   pz, u, v, w]';
    % process inputs to function
    px       = uu(7);       % inertial North position     
    py       = uu(8);       % inertial East position
    pz       = uu(9);
    u        = uu(10);       
    v        = uu(11);       
    w        = uu(12);       
    phi      = uu(6);       % roll angle         
    theta    = uu(5);       % pitch angle     
    psi      = uu(4);       % yaw angle     
    p        = uu(1);       % roll rate
    q        = uu(2);       % pitch rate     
    r        = uu(3);       % yaw rate    
    t        = uu(13);       % time

    % define persistent variables 
    persistent vehicle_handle;
    persistent Vertices
    persistent Faces
    persistent facecolors
    
    % first time function is called, initialize plot and persistent vars
    if t==0
        figure(1), clf
        [Vertices,Faces,facecolors] = defineVehicleBody(a,b,c);
        vehicle_handle = drawVehicleBody(Vertices,Faces,facecolors,...
                                               px,py,pz,phi,theta,psi,...
                                               [],'normal');
        title('Vehicle')
        xlabel('X')
        ylabel('Y')
        zlabel('Z')
        view(32,47)  % set the view angle for figure
        axis([-6,6,-6,6,-6,6]);
        hold on
        
    % at every other time step, redraw base and rod
    else 
        drawVehicleBody(Vertices,Faces,facecolors,...
                           px,py,pz,phi,theta,psi,...
                           vehicle_handle);
    end
end

  
%=======================================================================
% drawVehicle
% return handle if 3rd argument is empty, otherwise use 3rd arg as handle
%=======================================================================

function handle = drawVehicleBody(V,F,patchcolors,...
                                     px,py,pz,phi,theta,psi,...
                                     handle,mode)
  V = rotate(V, phi, theta, psi);  % rotate vehicle
  V = translate(V, px, py, pz);  % translate vehicle
  % transform vertices from NED to XYZ (for matlab rendering)
  %R = [...
  %    0, 1, 0;...
  %    1, 0, 0;...
  %    0, 0, -1;...
  %    ];
  %V = R*V;
  
  if isempty(handle)
  handle = patch('Vertices', V', 'Faces', F,...
                 'FaceVertexCData',patchcolors,...
                 'FaceColor','flat',...
                 'EraseMode', mode);
  else
    set(handle,'Vertices',V','Faces',F);
    drawnow
  end
end

%%%%%%%%%%%%%%%%%%%%%%%
function pts=rotate(pts,phi,theta,psi)

  % define rotation matrix (right handed)
  R_roll = [...
          1, 0, 0;...
          0, cos(phi), sin(phi);...
          0, -sin(phi), cos(phi)];
  R_pitch = [...
          cos(theta), 0, -sin(theta);...
          0, 1, 0;...
          sin(theta), 0, cos(theta)];
  R_yaw = [...
          cos(psi), sin(psi), 0;...
          -sin(psi), cos(psi), 0;...
          0, 0, 1];
  R = R_roll*R_pitch*R_yaw;  
    % note that R above either leaves the vector alone or rotates
    % a vector in a left handed rotation.  We want to rotate all
    % points in a right handed rotation, so we must transpose
  R = R';

  % rotate vertices
  pts = R*pts;
  
end
% end rotateVert

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% translate vertices by px, py, pz
function pts = translate(pts,px,py,pz)

  pts = pts + repmat([px;py;pz],1,size(pts,2));
  
end

% end translate


%=======================================================================
% defineVehicleBody
%=======================================================================
function [V,F,facecolors] = defineVehicleBody(a,b,c)

% Define the vertices (physical location of vertices
V = [...
    a/2, -b/2, -c/2;... %pt 1
    a/2, b/2, -c/2; ... %pt 2
    a/2, -b/2, c/2;... %pt 3
    a/2, b/2, c/2; ... %pt 4
    -a/2, -b/2, -c/2; ... %pt 5
    -a/2, b/2, -c/2;... %pt 6
    -a/2, -b/2, c/2;... %pt 7
    -a/2, b/2, c/2;... %pt 8
    ]';     
% % define faces as a list of vertices numbered above
  F = [...
        1, 2, 4, 3;...  % front top nose
        2, 4, 8, 6;...  % front left nose
        5, 6, 8, 7;...  % front bottom nose
        1, 3, 7, 5;...  % front right nose
        3, 7, 8, 4;...  % fueselage top
        1, 5, 6, 2;...  % fueselage left
        ];
% define colors for each face    
  myred = [1, 0, 0];
  mygreen = [0, 1, 0];
  myblue = [0, 0, 1];
  myyellow = [1, 1, 0];
  mycyan = [0, 1, 1];

  facecolors = [...
    myred;...       % nose
    mygreen;...     % fueselage
    myblue;...      % wing
    myyellow;...    % tail
    mycyan;...      % rudder
    myred;...      % rudder
    ];
end
  