function H=gaussian_filter(A,a,b);

r1=size(A,1);
r2=size(A,2);

aa=floor(a/2);
t=-aa:1:aa;
x=zeros(a,a);
y=zeros(a,a);

for i=1:a
    
    x(i,:)=t;
    y(:,i)=t;
    
end

G=(1/(2*pi*b^2))*(exp((-1/(2*b^2))*(x.^2+y.^2)));

g=fft2(G,r1,r2);
a=fft2(A);
M=a.*g;

H=ifft2(M);

H=mat2gray(H);
end