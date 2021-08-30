y = audioread("../../wav_files_coupe/ch7_coupe.wav");

% nbpar the order of the lpc coefficients
nbpar = 12;
% longeur de la fenêtre 
dfen = 735*2;
% longeur de saut
hop_length = 735;

duree = length(y);
nbfen = fix(duree/hop_length);
% initalisation of the matrix used to save lsp values
datalsf=zeros(nbpar,nbfen);

for k = 1:nbfen
    if  (k-1)*hop_length+dfen < duree
        ind = ((k-1)*hop_length+1:(k-1)*hop_length+dfen);
    else
        ind = ((k-1)*hop_length+1:duree);
    end
    % window hanning
    w = hanning(length(ind));
    sig = y(ind).*w;
    datalsf(1:nbpar,k) = poly2lsf(lpc(sig,nbpar));
    
end

