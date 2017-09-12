run ~/Desktop/rvctools/startup_rvc.m;
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
bot1 = SerialLink([L1,L2,L3,L4,L5,L6,L7],'name','Lab 1.1.1','base' , transl(0.024645, 0.219645, 0.118588)*trotz(pi/4)*transl(0.055695, 0, 0.011038));
bot2 = SerialLink([L1,L2,L3,L4,L5,L6,L7],'name','Lab 1.1.2','base' , transl(0.024645, 0.219645, 0.118588)*trotz(pi/4)*transl(0.055695, 0, 0.011038));
bot3 = SerialLink([L1,L2,L3,L4,L5,L6,L7],'name','Lab 1.1.3','base' , transl(0.024645, 0.219645, 0.118588)*trotz(pi/4)*transl(0.055695, 0, 0.011038));
bot4 = SerialLink([L1,L2,L3,L4,L5,L6,L7],'name','Lab 1.1.4','base' , transl(0.024645, 0.219645, 0.118588)*trotz(pi/4)*transl(0.055695, 0, 0.011038));
q1 = load('part1/part1_1.mat');
q1 = q1.q;
q2 = load('part1/part1_2.mat');
q2 = q2.q;
q3 = load('part1/part1_3.mat');
q3 = q3.q;
q4 = load('part1/part1_4.mat');
q4 = q4.q;

figure(1)
bot1.plot(q1)

figure(2)
bot2.plot(q2)

figure(3)
bot3.plot(q3)

figure(4)
bot4.plot(q4)

%% Calculate Transformation matrices for each of the 4 positions
T1_1 = bot1.fkine(q1);
T2_1 = bot1.fkine(q2);
T3_1 = bot1.fkine(q3);
T4_1 = bot1.fkine(q4);