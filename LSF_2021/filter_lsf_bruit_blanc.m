y = audioread("../../wav_files/chapiter7.wav");

dfen = 735;
% the white noise
u = randn(dfen,1);
w = hanning(dfen);
duree = 11723985;
nbfen = ceil(duree/dfen);

reconstructed_wav = zeros(duree,1);

for k = 1:nbfen
    ind = ((k-1)*dfen+1:k*dfen);
    reconstructed_wav(ind,1) = filter(1,lpc_predicted(k,:),u).*w;
    
end
figure(1)
plot(1:11723985, reconstructed_wav)
