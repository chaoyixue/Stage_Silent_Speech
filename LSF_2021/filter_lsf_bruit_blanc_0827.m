y = audioread("../../wav_files/chapiter7.wav");

dfen = 735*2;
hop_length = 735;
duree = 11720310;
nbfen = fix(duree/hop_length);

reconstructed_wav = zeros(duree,1);

for k = 1:nbfen-1
    if k == 1
        w = hanning(hop_length);
        u = randn(hop_length,1);
        ind = (1:hop_length);
        reconstructed_wav(ind,1) = filter(1,datalpc(1,:),u);
    else
        w = hanning(hop_length);
        u = randn(hop_length,1);
        ind = ((k-1)*hop_length+1:k*hop_length);
        lpc_fenetre = datalpc(k,:);
        reconstructed_wav(ind,1) = filter(1,lpc_fenetre,u);
    end
    
end
reconstructed_wav = reconstructed_wav/max(reconstructed_wav);
figure(1)
plot(1:11720310, reconstructed_wav)
audiowrite("filter_bruit_blanc_avec_lsf_origine_sans_fenetrage.wav", reconstructed_wav, 44100)
