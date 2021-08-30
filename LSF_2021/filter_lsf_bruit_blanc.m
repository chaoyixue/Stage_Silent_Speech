y = audioread("../../wav_files/chapiter7.wav");

dfen = 735;
% the white noise

w = hanning(dfen);
duree = 11720310;
nbfen = ceil(duree/dfen);

reconstructed_wav = zeros(duree,1);

for k = 1:nbfen
    u = 2*(rand(dfen,1)-0.5).*w;
    ind = ((k-1)*dfen+1:k*dfen);
    reconstructed_wav(ind,1) = filter(1,datalpc(k,:),u);
    
end
reconstructed_wav = reconstructed_wav/max(reconstructed_wav);
figure(1)
plot(1:11720310, reconstructed_wav)
audiowrite("filter_bruit_blanc_avec_lsf_origine.wav", reconstructed_wav, 44100)
