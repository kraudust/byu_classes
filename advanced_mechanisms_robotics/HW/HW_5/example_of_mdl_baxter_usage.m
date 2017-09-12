%calling the file as a function that returns a left and right arm
[left, right] = mdl_baxter('sim')

%defining an initial joint configuration
q0 = zeros(7,1)

%defining an initial joint velocity
qd0 = zeros(7,1)

%defining a test joint torque
tau = 10*ones(7,1)

%calculating the mass, corilios and gravity terms
M = left.inertia(q0.')
C = left.coriolis(q0.', qd0.')
G = left.gravload(q0.').'

%calculating the joint accleration given initital conditions and a torque -
q_dd = left.accel(q0.', qd0.', tau.')

%that last function is one that could be passed directly into ode45 to
%numerically integrate and calculate q and q_d given a torque profile over
%time.