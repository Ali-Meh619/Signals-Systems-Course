%%
%Q.1

m1=imread('download.jpg');
m1=rgb2gray(m1);
figure
subplot(1,2,1)
%m1=imnoise(m1,'gaussian',0.00001);
imshow(kirsch(m1))
xlabel('KIRSCH')
n=sobel(m1);
subplot(1,2,2)
imshow(n);
xlabel('SOBEL')



%%
%Q.4
clc
clear


subplot(1,2,1)
A1=imread('pic1.png');


aa1=imnoise(A1,'speckle',0.5);
aa2=imnoise(A1,'gaussian',0,0.05);
aa3=imnoise(A1,'poisson');
aa4=imnoise(A1,'salt & pepper');
imshow(aa4);
xlabel('SALT&PEPPER NOISE')
r1=size(A1,1);
r2=size(A1,2);

B=zeros(r1+100,r2+100);

B(51:r1+50,51:r2+50)=aa4;

A=B;

%A=aa2;


b=2.5;
a=5;
H=gaussian_filter(A,a,b);
subplot(1,2,2)
H1=H(51:r1+50,51:r2+50);
imshow(H1);

xlabel('FILTERED')
%%
%GAUSSIAN_FILTER & MEDIAN_FILTER
b=5;
figure
subplot(1,2,1)
imshow(aa4);
xlabel('SALT&PEPPER NOISE')

f=median_filter(aa4,b);
subplot(1,2,2)
imshow(f);
xlabel('FILTERED')


%%
%Q.3

m1=imread('pic1.png');
m2=imread('pic2.png');

f1=fft2(m1);
f2=fft2(m2);

a1=abs(f1);
a2=abs(f2);

p1=angle(f1);
p2=angle(f2);

f1_new=a2.*cos(p1)+i*(a2.*sin(p1));
k=ifft2(f1_new);
k=mat2gray(k);
    imshow(k)
        

%%
%Q.5 WITH MATLAB
clc
clear

load fmri.mat

h=imregcorr(image(:,:,1),image(:,:,2));

fixed = image(:,:,1);
moving = image(:,:,2);
figure
subplot(1,2,1)
imshow(fixed)
subplot(1,2,2)
imshow(moving)
figure

imshowpair(fixed, moving,'Scaling','joint');

[optimizer, metric] = imregconfig('multimodal');
optimizer.InitialRadius = 0.004;
optimizer.Epsilon = 1.5e-7;
optimizer.GrowthFactor = 1.001;
optimizer.MaximumIterations = 800;
tform = imregtform(moving, fixed, 'affine', optimizer, metric);
movingRegistered = imwarp(moving,tform,'OutputView',imref2d(size(fixed)));



figure
imshowpair(fixed, movingRegistered,'Scaling','joint')


u = [0 1]; 
v = [0 0]; 
[x, y] = transformPointsForward(tform, u, v); 
dx = x(2) - x(1); 
dy = y(2) - y(1); 
angle = -(180/pi) * atan2(dy, dax) ;

scale = 1 / sqrt(dx^2 + dy^2);
x_scale=scale*cos(angle);
y_scale=scale*sin(angle);




%%
%Q.5 MY OWN CODE
clc
clear

load fmri.mat

m1=image(:,:,1);

m2=image(:,:,2);


theta=(pi/180)*(15):pi/360:(25)*(pi/180);
b=10:0.5:20;
a=-15:0.5:-5;


oo=1;

for k=1:21
for j=1:21
for p=1:21

tform = affine2d([cos(theta(1,p)) -sin(theta(1,p)) 0; ...
                  sin(theta(1,p))  cos(theta(1,p)) 0; ...
                  a(1,k) b(1,j) 1]);
            
            
            m2= imwarp(m2,tform,'OutputView',imref2d(size(m1)));
                                      
            C=corr2(m1,m2);
            c(oo,1)=a(1,k);
            c(oo,2)=b(1,j);
            c(oo,3)=(180/pi)*theta(1,p);
            c(oo,4)=C;
            
           
            m2=image(:,:,2);
         oo=oo+1;
            
                    
           
end
oo=oo+1;
end
oo=oo+1;
end

[M,I] = max(c(:,4));

[I_row, I_col] = ind2sub(size(c),I);

tform = affine2d([cosd(c(I_row,3)) -sind(c(I_row,3)) 0; ...
                  sind(c(I_row,3))  cosd(c(I_row,3)) 0; ...
                  c(I_row,1) c(I_row,2) 1]);
            
            
            m2= imwarp(m2,tform,'OutputView',imref2d(size(m1)));
            
            
figure       
subplot(1,2,1)
imshow(m1);
subplot(1,2,2)
imshow(m2);

%%
%Q.2
clear;
clc;
img1 = imread('circles.jpg');
img1 = imresize(img1, 1); 
img1 = rgb2gray(img1);
img3 = img1;
img1 = double(img1);

img1 =edgee(img1, 0.05);

A = circles(img1,15,50);

img2=img3;

for i = 21:27
    b = reshape(A(i, :, :), size(A, 2), size(A, 3));
    [x, y] = find(b > 0.85);
    for j = 1:size(x, 1)
       img2 = d(img2, x(j), y(j), i); 
    end
end

imshow(img2);


%%
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
    
function     y=kirsch(A)


G(:,:,1)=[5 5 5;-3 0 -3;-3 -3 -3];
G(:,:,2)=[5 5 -3;5 0 -3;-3 -3 -3];
G(:,:,3)=[5 -3 -3;5 0 -3;5 -3 -3];
G(:,:,4)=[-3 -3 -3;5 0 -3;5 5 -3];
G(:,:,5)=[-3 -3 -3;-3 0 -3;5 5 5];
G(:,:,6)=[-3 -3 -3;-3 0 5;-3 5 5];
G(:,:,7)=[-3 -3 5;-3 0 5;-3 -3 5];
G(:,:,8)=[-3 5 5;-3 0 5;-3 -3 -3];

	r1 = size(A,1);
	r2 = size(A,2);
	
    a=fft2(A);
    
   for k=1:8
	g=fft2(G(:,:,k),r1,r2);
       
    s(:,:,k)=a.*g;
    
    S(:,:,k)=ifft2(s(:,:,k));
       
       
       
   end
   
   for j=1:size(S,2)
       
              for k=1:size(S,1)
                                 
           for i=1:8
   
               
               b(i,1)=S(k,j,i);
               
               
           end
              mag(k,j)=max(b);
              end
              end
   

 y=mat2gray(mag);
end

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


function f=median_filter(A1,b)



if mod(b,2)==1
   b=b+1; 
end

k=floor(b/2);



r1=size(A1,1);
r2=size(A1,2);

B=zeros(r1+2*k,r2+2*k);

B(k+1:k+r1,k+1:k+r2)=A1;

for i=k+1:k+r1
    for j=k+1:k+r2
        
        
        c(:,:)=B(i-2:i+2,j-2:j+2);
        
       a=median(c(:));
        
       f(i-k,j-k)=a;

        
        
    end
end

f=mat2gray(f);

end

function y = circles(img, min, maxx)

r1 = size(img, 1);
r2 = size(img, 2);
img2 = zeros(maxx, r1, r2);

for r = min:maxx
    
    kernel = zeros(2*r+1, 2*r+1);

    for t = 1:360
        x = r*cosd(t);
        y = r*sind(t);
        kernel(ceil(x)+r+1, ceil(y)+r+1) = 1 / r;
    end
    
    img1 = conv2(img, kernel, 'same');
    
    img2(r, :, :) = img1;
 
    
end

img2 = img2 / max(max(max(img2)));
y = img2;

end


function im =d(img, x, y, r)

im = img;
[xn, yn] = size(im);

for i = 1:360
    
    xd = ceil(x + r*cosd(i));
    yd = ceil(y + r*sind(i));
    
    if xd <= xn && yd <= yn
        im(max(xd, 1), max(yd, 1)) = 1;
    end
    
end

end

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
