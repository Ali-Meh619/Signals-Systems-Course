%% ford Fulkerson
clc;clear;
I = imread('img2.jpeg');
J1=imresize(I, 0.05);
I1=im2double(I);
i=(I1(:,:,1)+I1(:,:,2)+I1(:,:,3))/3;
J = imresize(i, 0.05);


rows=size(J,1);
columns=size(J,2);
[Y,X]=ndgrid(1:rows,1:columns);
x=X(:);
y=Y(:);
d=pdist2([x y],[x y]);
grey=double(J(:));
f=pdist2(grey,grey);
A=(0.01+f)./d;
A(f==0)=0.01./d(f==0);


A=A-diag(diag(A));
A(isnan(A))=0;;
A1=zeros(size(A,2)+2,size(A,2)+2);
A1(2:size(A,2)+1,2:size(A,2)+1)=A;
A1(1,2:size(A,2)+1)=(1/3);
A1(2:size(A,2)+1,1)=(1/3);
A1(size(A,2)+2,2:size(A,2)+1)=(2/3);
A1(2:size(A,2)+1,size(A,2)+2)=(2/3);

cap=A1;
%%
clc;clear;
Image = double(imread('img2.jpeg'));
I_R = imresize(Image(:,:,1),0.05);
rows=size(I_R,1);
columns=size(I_R,2);
N = rows*columns;
B = I_R / 255;
A = 1 - B;
A = A(:)';
B = B(:)';
QQ = reshape(I_R,1,[]);
p = zeros(N);
s = 1;

for i=1:N
    if ( (i+1)<=N && i ~= floor(i/columns)*columns )
        p(i,i+1) = filt(QQ(i),QQ(i+1),s); 
    end
    if  ( (i-1)>=1 && i ~= floor(i/columns)*columns+1 )
        p(i,i-1) = filt(QQ(i),QQ(i-1),s);
    end
    if ( (i-columns)>=1 )
        p(i,i-columns) = filt(QQ(i),QQ(i-columns),s);
    end
    if ( (i+columns)<=N )
        p(i,i+columns) = filt(QQ(i),QQ (i+columns),s);
    end
    end

cap = [0 A 0;
    zeros(rows*columns,1) p B';
    zeros(1,rows*columns+2)]; 
%%

%example

p=[0 1 2 4 0 2;
    0 0 0 4 5 0;
    0 0 0 0 9 1;
    0 0 0 0 0 10;
    0 0 0 0 0 8;
    0 0 0 0 0 0];


%%
ps = ST_cut(1,6,p,6);

a=graphh(ps);

%%
tic
ww=cap;
I_S = ST_cut(1,rows*columns+2,ww,rows*columns+2);
toc
%%
I_S(1,:) = [];
I_S(:,1) = [];
I_S(size(I_S),:) = [];
I_S(:,size(I_S)) = [];
[M,NN] = find(I_S ~= 0);
iL = zeros(rows,columns);
for o = 1:rows*columns
if find(M == o)
iL(o) = 200;
end
end
JJ=unit8(iL);
imshow(JJ);
%%
function o = filt(X,Y,sigma)
x=double(X(:));
y=double(Y(:));
o=exp((norm(-x+y))/sigma);
end

function cff=ST_cut(sS,sK,cap,nn)

cff=zeros(nn,nn);
mFX=0;mNc = [];
PT = bFS(sS,sK,cff,cap,nn);

while ~isempty(PT)
itt = inf;
kk=length(PT);
for i=1:kk-1
  
a=PT(i);
b=PT(i+1);
itt=min(itt, cap(a,b)-cff(a,b));
 end
 mNc =[mNc;find(itt == (cap - cff))];
 
 for i=1:kk-1
    a=PT(i);
    b=PT(i+1);
     
 cff(a,b)=cff(a,b)+itt;
 cff(b,a)=cff(b,a)-itt;
 end
 mFX=mFX+itt;
 PT = bFS(sS,sK,cff,cap,nn);
end
end

function Path = bFS(beG,tAR,clfw,cap,n)

    A1 =0;A2=1; A3=2;    
    head=1; tail=1;
    QW=[];Path=[];
    A(1:n)=A1;
    QW=[beG QW];
    A(beG)=A2;
    REp(beG) = -1;
    
REp=zeros(1,n);
while ~isempty (QW) 
 UU=QW(end);
 QW(end)=[];
   A(UU)=A3;
 for FQ=1:n               
if (A(FQ)==A1 && cap(UU,FQ)>clfw(UU,FQ) )
  QW=[FQ QW];
  A(FQ)=A2;
  REp(FQ)=UU;                        
   end
   end
   end
  if A(tAR)==A3      
  tOP=tAR;
  while REp(tOP)~=beG
   Path = [REp(tOP) Path]; 
    tOP=REp(tOP);
    end
    Path=[beG Path tAR];
    else
    Path=[];       
    end
end

function qqq=graphh(x)
    r=size(x,1);
    p=0;
    k=0;
    qqq=tril(x);
    for i=1:r
        for j=1:r
        if x(i,j)~=0
            p=p+1;
            s(1,p)=i;
            t(1,p)=j;
            weights(1,p)=x(i,j); 
        end
         if x(i,j)~=0 && i>j
            k=k+1;
            s1(1,k)=j;
            t1(1,k)=i;
            weights1(1,k)=x(i,j);    
        end
        end    
        end
    
    G = digraph(s,t,weights)
    G1=digraph(s1,t1,weights1)
    figure
    plot(G,'Layout','force','EdgeLabel',G.Edges.Weight)
    figure
    plot(G1,'Layout','force','EdgeLabel',G1.Edges.Weight) 
    end
   