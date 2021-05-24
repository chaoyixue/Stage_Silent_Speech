% read the audio file
[y,fs]=audioread("C:\Users\chaoy\Desktop\StageSilentSpeech\wav_files\chapiter1.wav");

% down sampling to 11025Hz which means a factor of 4
y_11025hz = downsample(y, 4); 

% nbpar the order of the lpc coefficients
nbpar = 12;

lpc_coefficients = lpc(y_11025hz,nbpar);
% calculation inversed
est_y = filter([0 -lpc_coefficients(2:end)],1, y_11025hz);
% plot the original signal and the signal reconstructed by lpc coefficients
plot(1:1849146,y_11025hz,1:1849146,est_y,'--')
grid
xlabel('Sample Number')
ylabel('Amplitude')
legend('Original signal','LPC estimate')

datalsp(1:nbpar) = poly2lsf(lpc(y_11025hz,nbpar));