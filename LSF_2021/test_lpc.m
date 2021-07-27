y = audioread("../../wav_files/chapiter7.wav");

% nbpar the order of the lpc coefficients
nbpar = 12;
dfen = 735;
w = hanning(dfen);
duree = length(y);
nbfen = ceil(duree/dfen);
% initalisation of the matrix used to save lsp values
datalpc=zeros(nbpar+1,nbfen);

for k = 1:nbfen
    % fix 朝0方向的取整
    ind = ((k-1)*dfen+1:k*dfen);
    % window hanning
    sig = y(ind).*w;
    datalpc(1:nbpar+1,k) = lpc(sig,nbpar);
end

