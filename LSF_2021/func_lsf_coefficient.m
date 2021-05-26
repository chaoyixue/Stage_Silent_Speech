function lsp_matrix = func_lsf_coefficient(filepath)
    % read the audio file
    [y,~]=audioread(filepath);
    % nbpar the order of the lpc coefficients
    nbpar = 12;
    dfen = 735;
    w = hanning(dfen);
    duree = length(y);
    nbfen = floor(duree/dfen);
    % initalisation of the matrix used to save lsp values
    lsp_matrix=zeros(nbpar+1,nbfen);
    
    for k = 1:nbfen
        % fix 朝0方向的取整
        ind = ((k-1)*dfen+1:k*dfen);
        % window hanning
        sig = y(ind).*w;
        lsp_matrix(1:nbpar,k) = poly2lsf(lpc(sig,nbpar));
    end
    

    

end