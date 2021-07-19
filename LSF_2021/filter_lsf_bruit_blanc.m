x = randn(703439100,1);
est_x = filter([0 -a(2:end)],1,x);

