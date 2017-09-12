d1 = 270.35/1000;
d3 = 364.35/1000;
d5 = 374.29/1000;
d7 = 229.525/1000;
a1 = 69/1000;
a3 = 69/1000;
a5 = 10/1000;

L1 = Link('revolute','d',d1,'a',a1,'alpha',-pi/2);
L2 = Link('revolute','d',0,'a',0,'alpha',pi/2,'offset',pi/2);
L3 = Link('revolute','d',d3,'a',a3,'alpha',-pi/2);
L4 = Link('revolute','d',0,'a',0,'alpha',pi/2);
L5 = Link('revolute','d',d5,'a',a5,'alpha',-pi/2);
L6 = Link('revolute','d',0,'a',0,'alpha',pi/2);
L7 = Link('revolute','d',d7,'a',0,'alpha',0);
bot = SerialLink([L1,L2,L3,L4,L5,L6,L7],'name','Lab 1.1.1','base' , transl(0.024645, 0.219645, 0.118588)*trotz(pi/4)*transl(0.055695, 0, 0.011038));


for i=1:1:10
    load(strcat('part3/part3_trial0',num2str(i-1),'.mat'));
    for j=1:1:length(q)
        xdot(i,j,:) = bot.jacob0(q(j,:))*q_dot(j,:)';
    end
end

for i=1:1:10
    for j=1:1:3
        figure (1)
        if j == 1
            plot(t,xdot(i,:,j),'b')
        elseif j == 2
            plot(t,xdot(i,:,j),'g')
        else
             plot(t,xdot(i,:,j),'r')
        end
                
        xlabel('Time (s)')
        ylabel('Velocity (m/s)')
        if i == 10
            legend('x','y','z')
        end
        hold on
        figure (2)
        if j == 1
            plot(t,xdot(i,:,j+3),'b')
        elseif j == 2
            plot(t,xdot(i,:,j+3),'g')
        else
             plot(t,xdot(i,:,j+3),'r')
        end
        xlabel('Time (s)')
        ylabel('Angular Velocity (rad/s)')
        if i == 10
            legend('w_x','w_y','w_z')
        end
        hold on
    end
end

figure (3)
for i=1:1:length(q)/50
    bot.plot(q(i*50,:))
end