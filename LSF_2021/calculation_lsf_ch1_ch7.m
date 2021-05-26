lsp_ch1 = func_lsf_coefficient("../../wav_files/chapiter1.wav");
ch1_reshape = lsp_ch1(:,1:10054);
lsp_ch2 = func_lsf_coefficient("../../wav_files/chapiter2.wav");
ch2_reshape = lsp_ch2(:,1:14441);
lsp_ch3 = func_lsf_coefficient("../../wav_files/chapiter3.wav");
ch3_reshape = lsp_ch3(:,1:8885);
lsp_ch4 = func_lsf_coefficient("../../wav_files/chapiter4.wav");
ch4_reshape = lsp_ch4(:,1:15621);
lsp_ch5 = func_lsf_coefficient("../../wav_files/chapiter5.wav");
ch5_reshape = lsp_ch5(:,1:14553);
lsp_ch6 = func_lsf_coefficient("../../wav_files/chapiter6.wav");
ch6_reshape = lsp_ch6(:,1:5174);
lsp_ch7 = func_lsf_coefficient("../../wav_files/chapiter7.wav");
ch7_reshape = lsp_ch7(:,1:15951);
lsp_all_chapiter = cat(2, ch1_reshape, ch2_reshape,ch3_reshape,...
ch4_reshape,ch5_reshape,ch6_reshape,ch7_reshape);
