chapiter_7_wav = audioread("chapiter7.wav");
load('energy_all_chapiter.mat')
load('energy_predicted.mat')
load('f0_all_chapiter.mat')
load('f0_predicted.mat')
load('lsf_predicted.mat')
load('lsp_all_chapiter.mat')
load('uv_all_chapiter.mat')
load('uv_predicted.mat')

original_lsf_ch7 = (lsp_all_chapiter(:,84679-15951+1:84679))';
original_f0_ch7 = (f0_original(:,84679-15951+1:84679))';
original_uv_ch7 = (uv_original(:,84679-15951+1:84679))';
original_energy_ch7 = (energy_original(:,84679-15951+1:84679))';

% between (n*10)s to (n+1)*10s of the chapiter 7
n = 4; % n*10 corrrespond to the current time
order_lsf = 5; % the order of lsf that we wanted to observe 
original_lsf_10s = original_lsf_ch7(n*600+1:(n+1)*600,order_lsf);
original_f0_10s = original_f0_ch7(n*600+1:(n+1)*600,:);
original_uv_10s = original_uv_ch7(n*600+1:(n+1)*600,:);
original_energy_10s = original_energy_ch7(n*600+1:(n+1)*600,:);
wav_10s = chapiter_7_wav(n*441000+1:(n+1)*441000,:);

% the predicted grandeurs
lsf_predicted_10s = lsf_predicted(n*600+1:(n+1)*600,order_lsf);
f0_predicted_10s = f0_predicted(n*600+1:(n+1)*600,:);
uv_predicted_10s = uv_predicted(n*600+1:(n+1)*600,:);
energy_predicted_10s = energy_predicted(n*600+1:(n+1)*600,:);

x_grandeur = (1:600)/60;
x_wav = (1:441000)/44100;
figure(5)
subplot(5,1,1)
plot(x_wav, wav_10s)
xlabel('time(s)')
title('the original wavform')
subplot(5,1,2)
plot(x_grandeur, original_lsf_10s)
hold on
plot(x_grandeur, lsf_predicted_10s,'r')
xlabel('time(s)')
legend('original lsf', 'predicted lsf')
title('the original lsf and the lsf predicted')
subplot(5,1,3)
plot(x_grandeur, original_f0_10s)
hold on
plot(x_grandeur, f0_predicted_10s,'r')
xlabel('time(s)')
legend('original f0', 'predicted f0')
title('the original f0 and the f0 predicted')
subplot(5,1,4)
plot(x_grandeur, original_uv_10s)
hold on
plot(x_grandeur, uv_predicted_10s,'r')
xlabel('time(s)')
legend('original uv', 'predicted uv')
title('the original voiced/unvoiced flags and the ones predicted')
subplot(5,1,5)
plot(x_grandeur, original_energy_10s)
hold on 
plot(x_grandeur, energy_predicted_10s,'r')
xlabel('time(s)')
legend('original energy', 'predicted energy')
title('the original energy and the predicted energy')