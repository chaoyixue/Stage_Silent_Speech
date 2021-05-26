% read the audio file
[y,fs]=audioread("C:\Users\chaoy\Desktop\StageSilentSpeech\wav_files\chapiter1.wav");

% nbpar the order of the lpc coefficients
nbpar = 12;
dfen = 735;
w = hanning(dfen);
duree = length(y);
nbfen = ceil(duree/dfen);
% initalisation of the matrix used to save lsp values
datalsp=zeros(nbpar+1,nbfen);

for k = 1:nbfen
    % fix 朝0方向的取整
    ind = ((k-1)*dfen+1:k*dfen);
    % window hanning
    sig = y(ind).*w;
    datalsp(1:nbpar,k) = poly2lsf(lpc(sig,nbpar));
end

% lpc_coefficients = lpc(y_11025hz,nbpar);
% calculation inversed
% est_y = filter([0 -lpc_coefficients(2:end)],1, y_11025hz);
% plot the original signal and the signal reconstructed by lpc coefficients
% plot(1:1849146,y_11025hz,1:1849146,est_y,'--')
% grid
% xlabel('Sample Number')
% ylabel('Amplitude')
% legend('Original signal','LPC estimate')
% 
% datalsp(1:nbpar) = poly2lsf(lpc(y_11025hz,nbpar));