function y=edgee(A,t)

Gx = [-1, 0, 1;-2, 0, 2;-1, 0, 1];
Gy = transpose(Gx);


    mf=fft2(A);
    r1=size(mf,1);
    r2=size(mf,2);
  
    gx=fft2(Gx,r1,r2);
    gy=fft2(Gy,r1,r2);
        
     s11=mf.*gx;
     s22=mf.*gy;
        
     s1=ifft2(s11);
     s2=ifft2(s22);


y = (s1.*s1 + s2.*s2).^(0.5);
a=y(:);
a_m=max(a);
y = y /a_m;
y = y - t;
y = ceil(y);
end
