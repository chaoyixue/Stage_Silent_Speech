lpc_predicted = zeros(15951,13);
for i = 1:15951
   lpc_predicted(i,:) = lsf2poly(lsf_predicted(i,:)); 
end

