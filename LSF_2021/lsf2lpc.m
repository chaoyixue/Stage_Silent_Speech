lpc_predicted = zeros(15946,13);
lsf_reforme = lsf_predicted(:,1:12);
for i = 1:15946
   lpc_predicted(i,:) = lsf2poly(lsf_reforme(i,:)); 
end

