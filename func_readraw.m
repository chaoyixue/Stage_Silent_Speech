function img = func_readraw(filepath, img_type)
    %% function used to read raw images and cut the raw image into bmp images
    %% parameters :
    %%      file_path : the path of the raw image file    
    %%      img_type : str , "levre", "langue"
    if strcmp(img_type, "levre")
        row = 744;
        col = 480; 
        fid = fopen(filepath, 'r');
        i=1
        while ~feof(fid)
            I = fread(fid, row*col, 'uint8=>uint8');
            img = reshape(I,row,col);
            img = img';
            % k=imshow(img);
            path_bmp = strcat("../data_2021/ch1_en/levre/", int2str(i), ".bmp");
            imwrite(img, path_bmp)
            i=i+1
        end
        
    elseif strcmp(img_type, "langue")
        row = 320;
        col = 240;
        
        fid = fopen(filepath, 'r');
        i=1
        while ~feof(fid)
            I = fread(fid, row*col, 'uint8=>uint8');
            img = reshape(I,row,col);
            img = img';
            % k=imshow(img);
            path_bmp = strcat("../data_2021/ch1_en/langue/", int2str(i), ".bmp");
            imwrite(img, path_bmp)
            i=i+1
        end
    end
    
end