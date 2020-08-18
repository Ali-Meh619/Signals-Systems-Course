function y=sobel(m)

    Gx=[-1 0 1; -2 0 2; -1 0 1];
    
    
	Gy=[-1 -2 -1; 0 0 0; 1 2 1];

   
    mf=fft2(m);
    r1=size(mf,1);
    r2=size(mf,2);
  
    gx=fft2(Gx,r1,r2);
    gy=fft2(Gy,r1,r2);
        
     s11=mf.*gx;
     s22=mf.*gy;
        
     s1=ifft2(s11);
     s2=ifft2(s22);
        
        k=sqrt(s1.^2+s2.^2);
       
y=mat2gray(k);
end
        