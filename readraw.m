%  row=576;  col=768;
%     fin=fopen('../data/raw/20200615_155926_RecFile_1/RecFile_1_20200615_155926_t3000_ultrasound_1_image.raw','r');
%     I=fread(fin,row*col,'uint8=>uint8'); 
%     Z=reshape(I,row,col);
%     Z=Z';
%     k=imshow(Z);
function readraw(rawfile_path)
     row=576;  col=768;
     fin=fopen(rawfile_path ,'r');
     I=fread(fin,row*col,'uint8=>uint8'); 
     Z=reshape(I,row,col);
     Z=Z';
     k=imshow(Z);
     
end