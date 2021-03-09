row = 744; col = 480;
fid = fopen('RecFile_1_20200615_155926_Reverse_1_output.raw','r');
i=1;
while ~feof(fid)
    I = fread(fid, row*col, 'uint8=>uint8');
    Z = reshape(I,row,col);
    Z = Z';
   % k=imshow(Z)
    i=i+1;
end
i
fclose(fid)