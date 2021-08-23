% the latest calculation is based on data cut
lsp_ch1 = func_lsf_coefficient("../../wav_files_coupe/ch1_coupe.wav");
%ch1_reshape = lsp_ch1(:,1:10054);
lsp_ch2 = func_lsf_coefficient("../../wav_files_coupe/ch2_coupe.wav");
%ch2_reshape = lsp_ch2(:,1:14441);
lsp_ch3 = func_lsf_coefficient("../../wav_files_coupe/ch3_coupe.wav");
%ch3_reshape = lsp_ch3(:,1:8885);
lsp_ch4 = func_lsf_coefficient("../../wav_files_coupe/ch4_coupe.wav");
%ch4_reshape = lsp_ch4(:,1:15621);
lsp_ch5 = func_lsf_coefficient("../../wav_files_coupe/ch5_coupe.wav");
%ch5_reshape = lsp_ch5(:,1:14553);
lsp_ch6 = func_lsf_coefficient("../../wav_files_coupe/ch6_coupe.wav");
%ch6_reshape = lsp_ch6(:,1:5174);
lsp_ch7 = func_lsf_coefficient("../../wav_files_coupe/ch7_coupe.wav");
%ch7_reshape = lsp_ch7(:,1:15951);
lsp_all_chapiter = cat(2, lsp_ch1, lsp_ch2, lsp_ch3, lsp_ch4, lsp_ch5, lsp_ch6, lsp_ch7);
% transpose to (84679,13)
lsp_all_chapiter = lsp_all_chapiter';
lsp_cut_all = lsp_all_chapiter;
save lsp_cut_all
