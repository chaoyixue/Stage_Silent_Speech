function img = func_readraw(filepath, img_type)
    if strcmp(img_type, "levre")
        row = 744;
        col = 480;  
    elseif strcmp(img_type, "langue")
        row = 320;
        col = 240; 
    end
    
    fid = fopen(filepath, 'r');
    i=1
    while ~feof(fid)
        I = fread(fid, row*col, 'uint8=>uint8');
        img = reshape(I,row,col);
        img = img';
        k=imshow(img);
        i=i+1
    end

end