I=imread('img1.jpg');
level=graythresh(I);
b=imbinarize(I,level);
imshowpair(I,b,'montage');
%%
I=imread('img2.jpeg');
level=graythresh(I);
b=imbinarize(I,level);
imshowpair(I,b,'montage');